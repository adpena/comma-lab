# Codex Findings: Compact VQ Pivot Work Orders

UTC: 2026-06-02T03:42:52Z

## Verdict

The HPRC spine bounded runner now treats the compact PACT/VQ mismatch verdict as an executable carrier-pivot signal, not just a demotion note. When every compact-base row is demoted by the compact-VQ audit, the runner selects zero demoted rows and opens a queue-owned work order for PR95/HNeRV, HiNeRV, and SNeRV carrier execution.

This keeps the score-lowering path focused on compact learned receivers after the measured PACT/VQ carrier failure. The current per-pair-latent VQ family remains demoted as a primary carrier until rebuilt as an RT/VQ residual-token bolt-on with measured value-per-byte evidence.

## Live Artifact

- Plan: `/Volumes/VertigoDataTier/pact/hprc_projection_gap_repairs/pact_nerv_vq_pact_nerv_vq_pvq_projection_gap_aa80443421cfbcb2/capacity_l16_e32_k64_ch48/hprc_spine_bounded_runner_plan_with_compact_vq_pivot_work_orders_20260602T034252Z.json`
- Selected demoted runner rows: `0`
- Compact carrier pivot work orders: `1`
- Pivot families: `pr95_hnerv`, `hi_nerv`, `snerv`
- Launch output root: `/Volumes/VertigoDataTier/pact/compact_carrier_pivots/compact_vq_pivot_30542d3214bfac78/`
- Safety: launch rows do not use `--overwrite`; output dirs must be empty before execution.
- Authority: false-authority only; exact CPU/CUDA dispatch remains blocked until byte-closed archive, receiver proof, full-video MLX replay, and exact-axis custody are present.

## Verification

- `python -m ruff check src/tac/substrates/hprc/spine_bounded_runner.py src/tac/substrates/hprc/tests/test_spine_bounded_runner.py` -> passed
- `python -m pytest src/tac/substrates/hprc/tests/test_spine_bounded_runner.py -q` -> 13 passed
- `python -m pytest src/tac/tests/test_compact_vq_pivot_audit.py src/tac/substrates/hprc/tests/test_spine_bounded_runner.py -q` -> 14 passed
- `git diff --check` -> passed

## Next Action

Execute the pivot rows in increasing spend:

1. PR95/HNeRV Stage-8 scorer-faithful continuation smoke, then full 600-pair if timing and custody are healthy.
2. HiNeRV QAT 600-pair compact learned receiver sweep under 178k/216k/285k ceilings.
3. SNeRV LF/HF receiver-proven advisory sweep only as a measured comparator.

Every survivor must emit the packet spine, archive bytes, receiver proof, full-video MLX value-per-byte profile, and exact blocker or dispatch packet. Do not spend more on primary PACT/VQ until a new design proves it can move score per byte.
