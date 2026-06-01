# Codex Findings: HPRC Direct MLX Cache Full-Video Bridge

UTC: 2026-06-01T01:05:58Z
Author: Codex
Axis: [macOS-MLX research-signal]
Authority: advisory only; no score claim; no promotion authority

## What Changed

The HPRC compact receiver now has a direct scorer-cache acquisition path for
local MLX sweeps. Instead of writing a 3.66 GB `0.raw` scratch file and then
reading it back into scorer tensors, the materializer can render
archive-contained HPRC receiver frames in chunks and stream them directly into
the SegNet/PoseNet cache tensors.

This is not an auth-eval shortcut. Receiver promotion still requires the
packaged `inflate.sh` proof and exact CPU/CUDA authority. The direct path is
local acquisition plumbing that removes avoidable raw-video I/O during candidate
search.

## Full-Video Evidence

Candidate:
`.omx/research/hprc_threshold_abs_le_3_receiver_proof_20260601T004446Z_codex/archive.zip`

Full 600-pair direct cache run:
`/Volumes/VertigoDataTier/pact/hprc_direct_cache_threshold_abs_le3_full600_20260601T010558Z`

Durable metadata copied to:
`.omx/research/hprc_direct_mlx_cache_full600_20260601T010558Z_codex/`

Results:

- cached pairs: `600`
- elapsed: `85.68786191940308` seconds
- raw scratch written: `false`
- direct raw-equivalent SHA-256:
  `ed484328d899fe6b7b0e1076db1d5d5ec38ac21429361122f96d7f93c8489e69`
- receiver-proof raw-output SHA-256:
  `ed484328d899fe6b7b0e1076db1d5d5ec38ac21429361122f96d7f93c8489e69`
- SHA match: `true`
- SegNet cache array hash:
  `5402540fa675956e513b6c39c7a8874124c37924a5b1b37efea486847403dfb4`
- PoseNet cache array hash:
  `fdbd43f092e3070054d4303929e8eee6ef8bbde1a5de0f02ae9bd19ed26e5c47`

## Integration

`tools/profile_hprc_mlx_component_neutralization.py` now defaults HPRC variant
cache materialization to `hprc-direct`, with `shell-inflate` retained as an
explicit parity/audit mode. This makes future residual-token sweeps faster by
default and prevents the direct path from becoming an orphaned helper.

## Blockers Preserved

- MLX-local cache rows remain false-authority.
- Direct cache rows cannot promote, rank, kill, or dispatch.
- Exact CPU/CUDA not executed.
- Receiver proof remains the archive/runtime custody surface.
