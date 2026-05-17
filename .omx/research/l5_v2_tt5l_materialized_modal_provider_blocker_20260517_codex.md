# L5 v2 TT5L materialized Modal provider blocker

- date: 2026-05-17
- scope: TT5L `random_lsb` materialized paired CPU/CUDA measurement
- provider: Modal
- failure class: `modal_workspace_billing_cycle_spend_limit_reached`
- score claim: `false`
- promotion eligible: `false`

## Attempted Work Unit

- archive: `experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_20260517_codex/random_lsb/archive.zip`
- archive sha256: `b6a5b63c0ea8acd582d8f273a1ee9e00f74becc9d1993a2f3085f2f89d64b1c7`
- archive bytes: `38911`
- run id: `l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_random_lsb_b6a5b63c0ea8`
- pair group: `pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda`
- canonical tool: `tools/dispatch_modal_paired_auth_eval.py`

## Result

The paired dispatch plan validated the archive SHA and produced distinct CPU/CUDA command plans for the corrected archive-specific output directories. Modal then failed before provider job spawn:

`App creation failed: workspace billing cycle spend limit reached`

No paired score job was created, no CPU/CUDA anchor exists for the `random_lsb` archive, and this is not TT5L method evidence. It is a provider-capacity blocker.

## Current Routing

`src/tac/optimization/l5_staircase_v2.py` now reads the JSON blocker artifact and changes the TT5L next action from retrying the same Modal dispatch to:

`resolve_l5_v2_tt5l_modal_provider_blocker_or_dispatch_alternate_provider`

The byte-closed TT5L work unit remains valid; only the Modal provider path is blocked until the billing limit is resolved or an alternate contest-compliant provider path is selected.
