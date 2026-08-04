# ddm_uf1 refresh registry receipt

Date: 2026-08-05. Arm: `uf1`. Scorer use: none. Protected files touched: none.

## Denominators

- quantities found: 12
- with consumers: 12
- with triggers: 12
- with known validity radius: 2
- scorer-free refreshed rows: 1
- exact invariant rows: 1
- heavy refreshes queued: 8
- fiber input blockers queued: 1

Typed registry: `refresh_registry.jsonl`.

## Executed scorer-free refresh

m66 gap decomposition was re-derived against qo1, using `.omx/research/ddm_sb1_20260804/sb1_rows.jsonl:1` and archive sha
`d5e814d5b9f65c3094b0e65fecdd7771734d03c420c63d1d2033a671b766986a`.

| component | value |
|---|---:|
| qo1 S | 0.7539807296911207 |
| PR130 floor S | 0.1721412974918964 |
| total gap | 0.5818394321992243 |
| seg gap | 0.4015190000000000 |
| pose gap | 0.0692658125616961 |
| rate gap | 0.1110546196375282 |
| seg share | 0.690086 |
| pose share | 0.119046 |
| rate share | 0.190868 |
| W bytes/flip | 1.2731082153320312 |

Axis warning: `CROSS-AXIS: ours='[macOS-CPU advisory]' vs floor='[contest-CUDA]'. The gap ranks axes correctly but its low-order digits are not a paired same-instrument comparison.`.

## Freshness-at-consumption guard

The qo1 consumer guard refused the fz4-only r9m advisory-to-contest calibration prior:

`r9m_advisory_to_contest_cpu_calibration_prior: QUEUED_HEAVY_REFRESH; route=full_recompute; trigger=archive sha moved fz4 -> qo1; projection is not consumed unchecked; owner=queue:next_modal_contest_cpu_row`

Current/invariant rows consumed by the guard: `m66_gap_decomposition_inputs_qo1`,
`W_bytes_per_flip_exchange`.

## Queued follow-ons

Heavy scorer/atlas refreshes are in `queued_refreshes.jsonl`. #891 fiber-transport input custody is
in `transport_refreshes.jsonl`; UF1 found no data-complete stale row with stored H_ab and mixed
partial inputs, so no transport was executed.

No score claim. No promotion eligibility. Own-vehicle frontier unchanged:
`S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`.
