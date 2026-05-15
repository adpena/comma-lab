# PR106 HDM8 Format-0x06 Sidecar Recode Refresh

Date: 2026-05-15

## Scope

Refresh the PR106 latent-sidecar recode profile against the current byte-closed
HDM8 implicit-length format-0x06 archive, not the stale 533-byte format-0x02
baseline.

This is a routing and hardening landing, not a new score claim.

## Source Artifact

- archive: `experiments/results/pr106_hdm8_implicit_len_fixed_meta_20260515_codex/emitted_candidates/pr101_implicit_len_fixed_meta_rank_elided_sidecar_format_0x06.archive.zip`
- archive_sha256: `44eb228ab2ae58f267d91c53db9119db3beb6a6c8488c7c903d9bb78e8798844`
- archive_bytes: `186382`
- exact_cuda_score: `0.2063530088582316`
- exact_cuda_axis: `[contest-CUDA]`
- source_sidecar_format_id: `0x06`
- source_sidecar_kind: `pr101_ranked_no_op_implicit_len_fixed_meta_rank_elided`
- current_charged_sidecar_bytes: `526`

## Tool Fix

`tools/profile_pr106_latent_sidecar_recode.py` previously refused to read a
format-0x06 archive as the source sidecar, even though the runtime and packet
compiler could already emit and consume that grammar. The profiler now decodes
implemented PR101-derived PR106 sidecar formats through
`decode_pr106_sidecar_packet_dim_delta(packet)` and reprofiles the semantic
`(dims, deltas)` arrays against the candidate set.

Regression added:

- `src/tac/tests/test_pr106_latent_sidecar_recode.py::test_recode_profile_tool_reads_implicit_len_fixed_meta_archive`

Verification:

- `.venv/bin/ruff check tools/profile_pr106_latent_sidecar_recode.py src/tac/tests/test_pr106_latent_sidecar_recode.py`
- `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_pr106_latent_sidecar_recode.py -q`

Result: `10 passed`.

## Refreshed Profile

- profile_json: `experiments/results/pr106_hdm8_fmt06_sidecar_recode_profile_20260515_codex/profile.json`
- profile_json_sha256: `ccffdef5f1006f357ad8e34b962b24fc31eb2d454d0acfd21130487a1de9a420`
- profile_md: `experiments/results/pr106_hdm8_fmt06_sidecar_recode_profile_20260515_codex/profile.md`
- profile_md_sha256: `0974e42d845067936da83e87e3609b89f354d7cdcdf3ae81601887aa37c4a4c5`

Implemented runtime candidates:

| candidate | sidecar_format_id | charged_sidecar_bytes | delta_vs_current |
|---|---:|---:|---:|
| `pr101_fixed_meta_rank_elided_sidecar_format_0x05` | `0x05` | 526 | 0 |
| `pr101_implicit_len_fixed_meta_rank_elided_sidecar_format_0x06` | `0x06` | 526 | 0 |
| `pr101_ranked_no_op_sidecar_format_0x02` | `0x02` | 533 | +7 |
| `current_pr100_dim_delta_brotli_q11` | `0x01` | 575 | +49 |

The current format-0x06 row carries both existing proof attachments:

- `runtime_consumption_claim: true`
- `full_frame_inflate_output_parity_claim: true`

## Classification

`score_claim=false`

The sidecar recode lane is saturated for the current lossless semantic grammar:
the best implemented candidates are tied at 526 charged sidecar bytes. Further
PR106 score movement requires either component-changing HDM/selector behavior,
decoder-context changes that remove more archive framing, or a new entropy model
with a runtime-consumed byte-closed packet. More lossless reprofiling of the
current `(dims, deltas)` stream should not be routed as expected score-lowering
work unless the candidate beats 526 charged sidecar bytes and has a runtime
consumption path.

