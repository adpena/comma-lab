# Codex Findings: Z8 Joint P18/P19 Executable Coefficient Waterfill

UTC: 2026-05-31T17:08:00Z
Author: Codex
Status: LANDED-IN-CODE, false-authority, exact-axis not claimed

## Finding

The Z8 600-pair advisory result is rate-bound, so the correct first executable
attack is not another SegNet-only repair row. The joint P18/P19 surface now
materializes into real Z8 Mallat coefficient bytes:

- pixel/pair joint weights are projected onto each wavelet detail subband;
- low joint-weight atoms are eligible only when the pose-null guard allows it;
- eligible detail coefficients are deterministically quantized and dead-zoned;
- the Z8HPC1 archive is rebuilt byte-closed;
- local receiver-resolution distortion and archive/wavelet byte deltas are
  measured and emitted as advisory metadata.

## Code Surfaces

- `tac.substrates.z8_hierarchical_predictive_coding.joint_coefficient_waterfill`
- `tools/materialize_z8_joint_p18_p19_deadzone_candidate.py`
- `pack_pair_pyramids_to_wavelet_blob(...)` public inverse in
  `canonical_quadruple_binding.py`
- Z8 joint variational contract now names the executable materializer.

## Authority Boundary

This is a local materializer/acquisition signal only. It writes byte-closed
archive variants, but it does not claim score, promotion, rank/kill authority,
or exact readiness until receiver proof and contest CPU/CUDA eval sign the
archive/runtime packet.

## Next Required Work

Run this materializer against the live Z8 600-pair archive with measured MLX
joint P18/P19 surfaces. If rate collapses without large receiver distortion,
promote the candidate to receiver proof and exact-axis handoff; if it does not,
append durable demotion evidence keyed by family/stage/scope.

## Local Smoke Anchor

Artifact:
`.omx/research/z8_joint_p18_p19_deadzone_smoke_20260531T1715Z/candidate/z8_joint_p18_p19_deadzone_manifest.json`

Input archive:
`experiments/results/z8_m11_l1_macos_cpu_mlx_local_end_to_end_smoke_canonical_evaluate_cpu_binding_20260530T161526Z/submission/archive/0.bin`

Configuration: uniform joint surface + all pose-null mask at 16x16 smoke
resolution, `joint_weight_quantile=1.0`,
`coefficient_deadzone_quantile=1.0`, `quantization_step=0.25`.

Observed local advisory result:

- Z8HPC1 bytes: 92,408 -> 6,625 (`archive_rate_ratio=0.0717`)
- wavelet blob bytes: 91,951 -> 5,831 (`wavelet_blob_rate_ratio=0.0634`)
- detail coefficients dead-zoned: 23,040 / 23,040
- small receiver distortion: `mse=0.0023139`, `mae=0.0286694`,
  `max_abs_delta=0.8287018`
- archive ZIP emitted:
  `.omx/research/z8_joint_p18_p19_deadzone_smoke_20260531T1715Z/candidate/archive.zip`
- authority: `[macOS-CPU advisory]`, no score claim, no exact dispatch,
  receiver proof not executed.

This is intentionally aggressive and distortion-heavy because it uses an all
pose-null/all-low-risk toy surface. Its value is proving the pipeline can turn
a joint P18/P19 surface into byte-closed Z8 coefficient mutation and measured
rate/distortion signal. The next meaningful run must use measured MLX joint
surfaces, not the uniform smoke mask.
