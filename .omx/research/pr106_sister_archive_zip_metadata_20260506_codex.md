# PR106 Sister Archive ZIP Metadata Determinism

Date: 2026-05-06
Agent: Codex
Evidence grade: empirical

## Finding

The PR106 stacked archive builder emitted deterministic ZIP member permissions,
but the three sister archive builders did not set `ZipInfo.external_attr` for
their single `0.bin` members:

- `experiments/build_pr106_latent_sidecar.py`
- `experiments/build_pr106_yshift_sidechannel.py`
- `experiments/build_pr106_lrl1_sidechannel.py`

These sister archives are the direct inputs to exact-eval candidate dispatch
and to `experiments/build_pr106_stacked.py`. Host-dependent permission metadata
would not change the logical payload, but it can change archive bytes and SHA
across platforms, which is unacceptable for reproducible custody.

## Change

All three sister builders now set the emitted `0.bin` ZIP member mode to
`0644` while preserving the fixed timestamp and stored compression.

## Guards

The following tests assert fixed member name, timestamp, compression, and mode:

- `src/tac/tests/test_pr106_latent_sidecar.py::test_cpu_smoke_builder_metadata_is_dispatch_fail_closed`
- `src/tac/tests/test_pr106_yshift_sidechannel.py::test_builder_writes_cross_platform_deterministic_zip_metadata`
- `src/tac/tests/test_pr106_lrl1_sidechannel.py::test_builder_zero_mode_metadata_is_fail_closed`

## Dispatch Status

No exact eval or remote GPU dispatch was attempted. This is archive-byte custody
hardening for the PR106 score-lowering sister lanes.
