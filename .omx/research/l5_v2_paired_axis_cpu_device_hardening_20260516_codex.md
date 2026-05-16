# L5 v2 paired-axis CPU device hardening - 2026-05-16

## Trigger

After hardening shared exact-eval custody, the L5-v2 paired-axis planning gate
still accepted any `contest_cpu` inflate/eval device string containing `cpu`.
That meant `macos-apple-m2-cpu` or `cpu-mps` could pass the paired-axis plan,
even though those are advisory/proxy hardware axes rather than contest-CPU
custody.

## Landing

`l5_staircase_v2` now uses the same positive/forbidden-token discipline for
paired-axis plan device fields:

- positive CPU/x86 token required;
- Apple/macOS/Darwin/ARM/MPS/Metal/MLX advisory tokens refused;
- negated tokens still fail closed.

## Tests

- `test_l5_v2_dispatch_readiness_rejects_macos_cpu_axis_semantics`

## Boundary

This gate is still planning/proof custody. It does not convert CPU evidence
into CUDA evidence and does not authorize dispatch or score claims.
