# Codex Findings: Z8 Entropy-Delta Campaign Plan

UTC: 2026-06-01T17:33:33Z
Axis: `[macOS-CPU advisory]`
Score claim: false
Promotion eligible: false

## Finding

The reusable Z8 detail headroom surface now has an automated bridge into the
byte-closed materializer path:

1. build full or sampled detail-coefficient headroom report;
2. solve the per-subband entropy-delta schedule under `max_subband_mse`;
3. emit a fail-closed materializer work order that consumes the schedule JSON.

New operator entrypoint:

```sh
uv run python tools/build_z8_entropy_delta_campaign_plan.py \
  --archive-bin <Z8HPC1 0.bin> \
  --output-dir <ssd artifact dir> \
  --num-pairs all \
  --quant-steps 0.00390625,0.0078125,0.015625,0.03125,0.0625,0.125,0.25 \
  --max-subband-mse 1e-5
```

This does not execute the materializer, receiver proof, replay, or auth eval.
It emits the exact command the queue runner should execute next, preserving
score authority as false until byte-closed replay and exact CPU/CUDA gates pass.

## Smoke Evidence

SSD smoke artifact:

- `/Volumes/VertigoDataTier/pact/z8_entropy_delta_campaign_plan_smoke_20260601T173333Z/manifest.json`

Real archive sampled:

- path: `experiments/results/z8_joint_p18_p19_deadzone_rate_attack/baseline/byte_closed_archive/0.bin`
- bytes: `152069787`
- sha256: `eb5d371c8d3c5afe0bff4b28b0ff98bc67bd562ea57d1aeef6b450c368ab9498`

Smoke settings:

- pairs: `1/600`
- quant steps: `0.0625`
- `allow_partial_headroom_coverage=true`
- `max_subband_mse=1e-3`

Result:

- `ready_for_queue_execution=true`
- blockers: `[]`
- materializer command emits a storage-layout-only v2 detail-entropy candidate
  via `tools/materialize_z8_joint_p18_p19_deadzone_candidate.py`

Strict full-campaign usage should keep partial coverage off and use `--num-pairs
all`; partial coverage is advisory only.

## Verification

Commands:

```sh
uv run pytest src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_entropy_delta_schedule.py src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_detail_coeff_entropy_headroom_report.py -q
uv run ruff check src/tac/substrates/z8_hierarchical_predictive_coding/entropy_delta_schedule.py src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_entropy_delta_schedule.py tools/build_z8_entropy_delta_campaign_plan.py
```

Result:

- `26 passed`
- `ruff`: all checks passed
