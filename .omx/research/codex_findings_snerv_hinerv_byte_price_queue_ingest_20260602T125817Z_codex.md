# Codex findings: SNeRV and HiNeRV byte-price queue ingest

## Landing

Built a combined false-authority NeRV rate/allocator queue with the corrected
SNeRV LF codec sweep and refreshed HiNeRV archive-size ladder as section-value
inputs:

- Queue: `nerv_rate_allocator_queue_20260602T125817Z_snerv_hinerv_byte_price.json`
- SNeRV source: `snerv_lf_payload_codec_sweep_20260602T125202Z.json`
- HiNeRV source: `hinerv_archive_size_ladder_20260602T125249Z.json`

The queue is now the shared consumer-facing surface for final-rate attack,
bit allocator, and bounded-runner planning. It does not launch dispatch, exact
eval, full-video eval, promotion, or rank/kill.

## Section Admission Summary

- Section-admission plans: 2
- Section-admission rows: 10
- Decisions: `demote=10`
- Source schemas:
  - `snerv_lf_payload_codec_sweep.v1`
  - `hinerv_archive_size_ladder.v1`

## SNeRV Rows

The corrected tiny SNAR packet sweep priced valid LF codec candidates but kept
them demoted because scorer replay, full-video archive replay, and contest
CPU/CUDA evidence are missing.

- `int64_lzma`: byte delta 0, economic decision `protect`, final `demote`
- `delta_varint`: byte delta +241, economic decision `protect`, final `demote`
- `portfolio_auto`: byte delta +241, economic decision `protect`, final `demote`
- `zero_run`: byte delta +293, economic decision `protect`, final `demote`
- `int2`/`int4`/`int8`: exact signed range refused on this packet, final `demote`

## HiNeRV Rows

HiNeRV modelsize increments are priced as candidate byte spends and remain
demoted until decoder-weight saliency/scorer replay supplies non-rate deltas:

- tiny -> small: +112,628 B, requires 0.07499436217224392 non-rate drop
- small -> base: +150,468 B, requires 0.10019046495838689 non-rate drop
- base -> wide: +414,354 B, requires 0.2759013206619842 non-rate drop

## Verification

- Focused tests: `21 passed`
- Ruff: clean
- Py compile: clean

## Authority

False-authority only. The queue is a planning/control artifact, not a score
claim and not exact-dispatch-ready.

## Next Work

Use this same queue path after the next full-600 SNeRV SNAR packet and after
HiNeRV scorer replay emits decoder-weight non-rate deltas. The controller can
then cut, protect, admit, or retrain sections based on the fixed contest byte
price instead of ad hoc rate intuition.
