# HiNeRV epoch7749 hard-pair hitlist

Axis: `[macOS-MLX research-signal]`, false-authority only.

## Inputs

- MLX response with per-pair components: `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_epoch7749_full600_mlx_replay_20260604T005700Z/mlx_response_gpu_full600_components.json`
- Pose component array SHA-256: `339c046859e018cffe7ac4302e926f8c859001b9dca40c22356270d08e50caed`
- Seg component array SHA-256: `3da3d3cfc0f911bb0400caaea209684eb4d4f1dd6ba5d72994b834a119ee99e3`
- Source archive SHA-256: `d9191f19a99cf33846821806ed2b64aa0027b25ccd05afa9e45ddc69ad67224e`
- Source archive bytes: `121572`

## Hitlist

- Output: `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_epoch7749_full600_mlx_replay_20260604T005700Z/hard_pair_hitlist_top16pct_v2.json`
- Schema: `nerv_hard_pair_hitlist.v1`
- Pair count: `600`
- Top K: `96`
- Ranking formula: `100*segnet_distortion + 5/sqrt(10*avg_posenet_dist)*posenet_distortion`
- First pair rows: `90,523,1,54,14,0,42,21,16,47`

## Guardrail

The scorer cache `pair_indices.npy` stores source frame pairs, e.g. `[180,181]`,
not training pair-row IDs. The hard-pair hitlist must emit `pair_indices` in
runner coordinates `0..599` and preserve source frame pairs only as provenance.
`src/tac/adaptation/hard_pair_hitlist.py` now enforces that distinction.

## Next Action

Use this file as `--prioritized-pair-indices-file` for the next HiNeRV successor
run. The prior epoch7749 replay is rate-valid but fit-invalid, so the successor
should preserve the compact archive grammar and concentrate training pressure
on these scorer-marginal hard pairs rather than exact-gating the current packet.
