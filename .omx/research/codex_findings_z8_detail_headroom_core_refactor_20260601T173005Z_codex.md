# Codex Findings: Z8 Detail Headroom Core Refactor

UTC: 2026-06-01T17:30:05Z
Axis: `[macOS-CPU advisory]`
Score claim: false
Promotion eligible: false

## Finding

The Z8 detail-coefficient entropy headroom path was already mathematically useful
but lived in `tools/z8_detail_coeff_entropy_headroom_report.py`, making queue and
allocator consumers reach into a CLI script. I moved the reusable byte/entropy
math into:

- `src/tac/substrates/z8_hierarchical_predictive_coding/detail_entropy_headroom.py`

The tool is now a thin CLI adapter over that package module. This keeps the
existing report schema stable while making the headroom surface importable by
queue-owned Z8/P18/P19/P11/P15 campaign runners.

## Smoke Evidence

Real archive sampled:

- path: `experiments/results/z8_joint_p18_p19_deadzone_rate_attack/baseline/byte_closed_archive/0.bin`
- bytes: `152069787`
- sha256: `eb5d371c8d3c5afe0bff4b28b0ff98bc67bd562ea57d1aeef6b450c368ab9498`

SSD artifact:

- `/Volumes/VertigoDataTier/pact/z8_detail_headroom_core_smoke_20260601T173005Z/manifest.json`

One-pair advisory result at `Delta=0.0625`:

- current detail bytes: `254058`
- v2 codec detail bytes: `10054`
- structured floor bytes: `9414`
- headroom fraction: `96.0%`
- mean distortion MSE: `2.554e-04`

This confirms the core refactor did not lose the existing Z8 rate-axis signal:
the detail blob remains the archive-rate hotspot, and the live quantized
subband codec is still close to the structured floor at this operating point.

## Verification

Commands:

```sh
uv run pytest src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_detail_coeff_entropy_headroom_report.py src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_entropy_delta_schedule.py -q
uv run ruff check src/tac/substrates/z8_hierarchical_predictive_coding/detail_entropy_headroom.py tools/z8_detail_coeff_entropy_headroom_report.py src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_detail_coeff_entropy_headroom_report.py src/tac/substrates/z8_hierarchical_predictive_coding/entropy_delta_schedule.py
```

Result:

- `24 passed`
- `ruff`: all checks passed

## Next Integration

The queue runner should call the package module directly, then feed the report
into `entropy_delta_schedule.build_entropy_delta_schedule_from_headroom_report`
and `build_entropy_delta_materializer_work_order`. Exact score movement still
requires byte-closed materialization, receiver proof, local replay, and then
CPU/CUDA auth gates.
