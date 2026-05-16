# exact-eval contest CPU custody hardening - 2026-05-16

## Trigger

L5/Cathedral adversarial review found that `contest_cpu` evidence could pass
the shared custody validator with macOS/Apple CPU advisory runs, as long as the
axis and device fields carried generic `cpu` tokens. That can corrupt paired
promotion and L5-v2 architecture-lock gates by mixing `[macOS-CPU advisory]`
with `[contest-CPU]`.

## Landing

Commit target: `src/tac/exact_eval_custody.py`.

The validator now treats `contest_cpu` as a contest custody class, not merely
"not CUDA":

- hardware must carry a positive CPU/x86 token;
- Apple/macOS/Darwin/ARM/MPS/Metal/MLX advisory tokens are refused in hardware;
- CPU inflate/eval device strings carrying forbidden advisory tokens are refused;
- recognized auth-eval commands carrying forbidden advisory tokens are refused.

Linux/x86 cloud CPU evidence remains valid. Modal or other providers can still
produce valid CPU-axis evidence when the recorded hardware/device/command
custody reflects contest-compliant CPU execution rather than local macOS/MPS.

## Tests

- `test_validate_exact_eval_evidence_accepts_linux_x86_cpu_axis`
- `test_validate_exact_eval_evidence_rejects_macos_cpu_advisory_axis`

## Follow-up

Keep `[contest-CPU]`, `[contest-CUDA]`, and `[macOS-CPU advisory]` separate in
reports and L5-v2 paired gates. Do not promote CPU/CUDA deltas across hardware
or runtime axes without matching custody.
