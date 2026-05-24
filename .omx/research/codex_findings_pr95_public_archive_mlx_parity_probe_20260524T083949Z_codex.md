# Codex Findings: PR95 Public Archive MLX Parity Probe

UTC: 2026-05-24T08:39:49Z
Lane: `lane_pr95_hnerv_mlx_reproduction`
Evidence grade: `[macOS-MLX research-signal]`

## What Landed

- Added a native PR95 public-archive packet parser in `tac.local_acceleration.pr95_hnerv_mlx`.
- Added an MLX-vs-PyTorch forward-parity helper that consumes the actual public PR95 `archive.zip/0.bin` packet bytes.
- Added `tools/probe_pr95_public_archive_mlx_parity.py` to emit durable custody/parity JSON.
- Added regression coverage for packet parsing, byte custody, false-authority fields, and MLX CPU forward parity.

## Live Artifact

Parity artifact:

`experiments/results/pr95_public_archive_mlx_parity_20260524T083932Z/forward_parity_probe.json`

Export round-trip artifact:

`experiments/results/pr95_public_archive_export_roundtrip_20260524T084706Z/export_summary.json`

Runtime-consumption proof artifact:

`experiments/results/pr95_native_export_runtime_consumption_20260524T085523Z/runtime_consumption_proof.json`

Public packet custody:

- Archive ZIP SHA-256: `e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a`
- ZIP member: `0.bin`
- Member bytes: `178309`
- Member SHA-256: `4b8013fb1168e21b12fc7ee6c395e032660d88daded48a0b845085e7d5eb11c4`
- Parsed meta: `n_pairs=600`, `latent_dim=28`, `base_channels=36`, `eval_size=[384, 512]`
- Parsed decoder tensors: `28`
- Parsed latents: `[600, 28]`

Forward-parity probe on latent row `0`:

| MLX device | passed | max_abs | mean_abs |
| --- | ---: | ---: | ---: |
| `cpu` | true | `0.0011749267578125` | `0.00004146706487517804` |
| `gpu` | false | `1.5164718627929688` | `0.04426033794879913` |

## Interpretation

This closes the previous synthetic-only parity gap for the CPU implementation path:
the MLX port can now prove source-packet forward parity against the actual public
PR95 archive bytes on local MLX CPU.

It also isolates a real MLX GPU drift surface on the same packet. That drift is
implementation/calibration signal only. It is not a score, not promotion
authority, not rank/kill authority, and not exact-dispatch readiness.

The archive export path now also round-trips the public PR95 packet grammar:
state + latents -> source-compatible `0.bin` -> deterministic stored ZIP ->
native parser. The rebuilt member SHA matches the original public `0.bin`
member SHA (`4b8013fb1168e21b12fc7ee6c395e032660d88daded48a0b845085e7d5eb11c4`);
the rebuilt ZIP SHA differs because the deterministic writer normalizes ZIP
container metadata.

The native-export runtime proof now exercises the actual public PR95
`inflate.sh` contract on a one-pair native-built packet:

- Runtime proof: `runtime_consumption_proven=true`
- Expected raw bytes: `6104016`
- Actual raw bytes: `6104016`
- Raw SHA-256: `3f672470f2142b7eeee0e20082a29e46e8c7d3355e696eb7b0be3fe40cab188f`

This proves runtime consumption for the PR95 grammar/export path. It is still a
small runtime smoke, not full public-packet inflate parity and not score
authority.

## False-Authority Boundary

Every emitted packet and result carries:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- exact-readiness blockers requiring full-frame inflate/runtime consumption proof
  and exact CPU/CUDA auth eval before any score claim.

## Verification

- `.venv/bin/ruff check src/tac/local_acceleration/pr95_hnerv_mlx.py src/tac/tests/test_pr95_hnerv_mlx.py tools/probe_pr95_public_archive_mlx_parity.py`
- `.venv/bin/python -m pytest src/tac/tests/test_pr95_hnerv_mlx.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_optimizer_scheduler_registry.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_run_pr95_mlx_timing_smoke_plan.py src/tac/tests/test_pr95_hnerv_mlx.py -q` (`24 passed`)
- `.venv/bin/python tools/probe_pr95_public_archive_mlx_parity.py --mlx-device cpu --mlx-device gpu --sample-indices 0 --output-json experiments/results/pr95_public_archive_mlx_parity_20260524T083932Z/forward_parity_probe.json`
- Native export round trip to `experiments/results/pr95_public_archive_export_roundtrip_20260524T084706Z/export_summary.json`
- Native one-pair runtime proof with `tools/prove_pr95_public_archive_runtime_consumption.py` to `experiments/results/pr95_native_export_runtime_consumption_20260524T085523Z/runtime_consumption_proof.json`
- `.venv/bin/python tools/lane_maturity.py validate`

## Next Engineering Step

Use the packet parser/exporter/runtime proof to build a byte-closed PR95 queue
bridge for locally trained MLX candidates:

1. Export MLX state and latents into PR95 `0.bin` grammar.
2. Run same-runtime `inflate.sh` output proof on a tiny file list.
3. Feed candidate runtime/profile rows into the optimizer queue as non-promotable
   local training signal until exact CPU/CUDA auth gates land.
