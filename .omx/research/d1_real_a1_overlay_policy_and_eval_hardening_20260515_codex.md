# D1 Real-A1 Overlay Policy Product + Eval Hardening - 2026-05-15

## Scope

Continue D1 score-lowering work without treating synthetic smoke packets as
contest candidates.

## Findings

- Prior D1 policy-product materialization used the local smoke payload whose
  `a1.bin` bytes were `smoke_base_archive_bytes`; those archives are runtime
  and materializer proofs only, not exact-eval candidates.
- The harvested Modal D1 run
  `substrate_d1_segnet_margin_polytope_modal_t4_dispatch_20260514T134005Z__smoke__100ep`
  reached real 600-pair D1 construction and successful inflate, then failed in
  upstream CUDA evaluation with DALI/NVML before any score JSON was produced.
- The harvested real D1 sidecar has:
  - `d1_polytope.bin` sha256:
    `a25ae4d29fe9ea4e380e5c3e1992c1f45eb8510719ffc4c04470d2ac8c4691c6`
  - A1 inner payload sha256:
    `8e664385af0a25ec98bd02d97b697fbf0d2bb3c2d954f5aa5c95b5131330a243`

## Code Hardening

- `experiments/contest_auth_eval.py` now sets `DALI_DISABLE_NVML=1` at the
  canonical upstream-evaluate subprocess boundary. This protects D1 and every
  other Modal/T4 exact-eval path from rediscovering the same NVML failure in a
  lane-local script.
- `build_readiness_manifest(...)` now accepts `base_archive_evidence_grade`.
  D1 smoke passes `synthetic_smoke`, so synthetic-base packets stay
  `ready_for_exact_eval_dispatch=false` even when the overlay runtime is
  operational.

## Real-A1 Candidate Materialization

Command:

```bash
PYTHONPATH=src .venv/bin/python tools/build_d1_overlay_policy_candidates.py \
  --d1-bin experiments/results/lane_substrate_d1_segnet_margin_polytope_modal_t4_dispatch_20260514T134005Z__smoke__100ep_modal/harvested_artifacts/d1_polytope.bin \
  --a1-bin experiments/results/lane_substrate_d1_segnet_margin_polytope_modal_t4_dispatch_20260514T134005Z__smoke__100ep_modal/harvested_artifacts/a1.bin \
  --output-dir experiments/results/d1_overlay_policy_product_candidates_real_a1_20260515_codex \
  --policies rgb,green,neg_green,rb_pos_g_neg \
  --amplitude-scales 0.5,1.0 \
  --sign-policies payload,negate_payload,alternating_pairs
```

Output manifest:

`experiments/results/d1_overlay_policy_product_candidates_real_a1_20260515_codex/d1_overlay_policy_candidates_manifest.json`

Materialized 24 real-A1 candidates. Smallest byte rows:

| candidate | bytes | archive sha256 |
| --- | ---: | --- |
| `d1_overlay_channel_rgb_amp_1_sign_payload` | 221666 | `fe4191c6c86d690d22cb2bb749e354e0f7750eb5e6a037aff9b262ed11ceeb3a` |
| `d1_overlay_channel_rgb_amp_0p5_sign_payload` | 221668 | `611b078396fafbf832bfddf42130d067f8aced251975d80fe9eb69914389a54d` |
| `d1_overlay_channel_green_amp_1_sign_payload` | 221668 | `7728aba0c9dd72fa9c6364574050f5426af76a478d8f5c701cfbb4ef42141a5a` |
| `d1_overlay_channel_green_amp_0p5_sign_payload` | 221670 | `464db0d332101ef80ee997f9605c3ad949c30c9cf77395c9d5ce444f181c0494` |

## Evidence Status

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false` in candidate manifests until paired
  `[contest-CUDA]` and `[contest-CPU]` evals land with archive/runtime custody.
- Next score-bearing action: paired Modal auth eval of a tiny subset, starting
  with `d1_overlay_channel_rgb_amp_1_sign_payload` as the byte-minimal
  real-A1 D1 policy candidate.

## Paired Eval Dispatch

Candidate:

- id: `d1_overlay_channel_rgb_amp_1_sign_payload`
- archive bytes: `221666`
- archive sha256:
  `fe4191c6c86d690d22cb2bb749e354e0f7750eb5e6a037aff9b262ed11ceeb3a`
- pair group: `d1_real_a1_rgb_amp1_payload_pair_20260515T1745Z`

First dispatch attempt
`d1_real_a1_rgb_amp1_payload_pair_20260515T1735Z` failed fast before auth eval:
the paired planner passed the full local `submission_dir/inflate.sh` path while
the Modal wrapper expects an inflate path relative to uploaded `--submission-dir`.
Fix landed in `tools/dispatch_modal_paired_auth_eval.py`; paired plans now
normalize this to `inflate.sh`.

Retried paired dispatch:

| axis | lane | Modal call id | status at 2026-05-15T17:35Z |
| --- | --- | --- | --- |
| `[contest-CUDA]` | `d1_real_a1_rgb_amp1_payload_contest_cuda` | `fc-01KRPB8K89XRD3Y9FMSJ099MG2` | pending recovery |
| `[contest-CPU]` | `d1_real_a1_rgb_amp1_payload_contest_cpu` | `fc-01KRPB945WH0R90KHAR7Z1C3VW` | pending recovery |

Recovery commands:

```bash
.venv/bin/python tools/recover_modal_auth_eval.py \
  --output-dir experiments/results/modal_auth_eval/d1_real_a1_rgb_amp1_payload_paired_auth_20260515T1745Z_cuda

.venv/bin/python tools/recover_modal_auth_eval.py \
  --output-dir experiments/results/modal_auth_eval_cpu/d1_real_a1_rgb_amp1_payload_paired_auth_20260515T1745Z_cpu
```

## Paired Eval Recovery - 2026-05-15T17:37Z

Both paired Modal calls recovered terminal pre-score failures. No
`contest_auth_eval.json` was produced, so there is no score claim and no
promotion evidence.

| axis | call id | terminal status | root cause |
| --- | --- | --- | --- |
| `[contest-CUDA]` | `fc-01KRPB8K89XRD3Y9FMSJ099MG2` | `failed_modal_auth_eval_no_score_claim` | D1 inflate no-op guard: `pairs_modified=0`, `bytes_changed=0` |
| `[contest-CPU]` | `fc-01KRPB945WH0R90KHAR7Z1C3VW` | `failed_modal_cpu_auth_eval_no_score_claim` | same D1 inflate no-op guard |

Recovery artifacts:

- CUDA:
  `experiments/results/modal_auth_eval/d1_real_a1_rgb_amp1_payload_paired_auth_20260515T1745Z_cuda/modal_auth_eval_recover_summary.json`
- CPU:
  `experiments/results/modal_auth_eval_cpu/d1_real_a1_rgb_amp1_payload_paired_auth_20260515T1745Z_cpu/modal_auth_eval_recover_summary.json`

The failure is classified as `archive/runtime bug: decoded zero D1 overlay`,
not a D1 method negative. The candidate carried 43 KB of D1 sidecar bytes but
the decoded overlay lattice was all zero.

## D1 Xray Root Cause

Canonical xray command:

```bash
PYTHONPATH=src .venv/bin/python tools/xray_d1_overlay_payload.py \
  --d1-bin experiments/results/d1_overlay_policy_product_candidates_real_a1_20260515_codex/d1_overlay_channel_rgb_amp_1_sign_payload/submission_dir/d1_polytope.bin \
  --json-out experiments/results/d1_overlay_policy_product_candidates_real_a1_20260515_codex/d1_overlay_channel_rgb_amp_1_sign_payload/d1_overlay_xray_20260515.json
```

Xray result:

- `decoded_noise_nonzero_pixels=0 / 196608`
- `camera_overlay_nonzero_pixels=0`
- `estimated_changed_bytes_upper_bound_per_pair=0`
- `integer_feasible_pixels=0`
- `max_safe_budget_lsb=0.38812318444252014`
- `mean_safe_budget_lsb=0.28066444396972656`
- blockers:
  - `d1_decoded_polytope_payload_all_zero`
  - `d1_camera_overlay_all_zero`
  - `d1_overlay_all_zero_after_attenuation`
  - `d1_estimated_changed_bytes_upper_bound_zero`
  - `d1_no_integer_feasible_pixels_under_lipschitz_bound`

Interpretation: with the current archived D1 setting `jacobian_lipschitz=20`,
the certified safe budget never reaches one uint8 LSB, so the integer overlay
cannot modify frames under the stated bound. This explains why the original
Modal run inflated successfully under the older runtime but carried dead rate:
it did not fail closed on `bytes_changed=0`.

## Additional Hardening Landed

- New reusable diagnostics module:
  `src/tac/substrates/d1_segnet_margin_polytope/diagnostics.py`
- New operator xray CLI:
  `tools/xray_d1_overlay_payload.py`
- `tools/build_d1_overlay_policy_candidates.py` now records
  `d1_overlay_diagnostics` and embeds decoded-zero/integer-infeasible blockers
  in every candidate manifest.
- `build_readiness_manifest(...)` now accepts decoded-noise, camera-overlay,
  and integer-feasible pixel counts; exact-eval readiness is blocked when these
  prove a dead-rate or uncertified integer-overlay packet.
- D1 full training now computes these diagnostics immediately after sidecar
  roundtrip and before auth-eval gating, preventing another D1 dead-rate
  dispatch.

Verification:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  src/tac/substrates/d1_segnet_margin_polytope/tests \
  src/tac/tests/test_build_d1_overlay_policy_candidates.py \
  src/tac/tests/test_xray_d1_overlay_payload.py

ruff check \
  src/tac/substrates/d1_segnet_margin_polytope/diagnostics.py \
  src/tac/substrates/d1_segnet_margin_polytope/__init__.py \
  src/tac/substrates/d1_segnet_margin_polytope/archive.py \
  experiments/train_substrate_d1_segnet_margin_polytope.py \
  tools/build_d1_overlay_policy_candidates.py \
  tools/xray_d1_overlay_payload.py \
  src/tac/substrates/d1_segnet_margin_polytope/tests/test_d1_substrate.py \
  src/tac/tests/test_build_d1_overlay_policy_candidates.py \
  src/tac/tests/test_xray_d1_overlay_payload.py
```

Results: `121 passed`; ruff clean; `git diff --check` clean.

Next D1 score-lowering actions:

1. Re-encode D1 from the same real margin map with a calibrated `L` sweep that
   has nonzero `integer_feasible_pixels`, then run the same xray before any
   auth eval.
2. Prefer shrunk margin-map resolution `(96, 128)` once the overlay is nonzero;
   full-resolution `384x512` paid 43 KB before any effect.
3. Pair each surviving D1 candidate across `[contest-CUDA]` and
   `[contest-CPU]`; do not promote either axis from the other.

## Real-A1 Payload-Budget Sweep - 2026-05-15

The D1 policy materializer now supports payload regeneration from the harvested
real margin map, not only metadata policy mutation.

Command:

```bash
PYTHONPATH=src .venv/bin/python tools/build_d1_overlay_policy_candidates.py \
  --d1-bin experiments/results/lane_substrate_d1_segnet_margin_polytope_modal_t4_dispatch_20260514T134005Z__smoke__100ep_modal/harvested_artifacts/d1_polytope.bin \
  --a1-bin experiments/results/lane_substrate_d1_segnet_margin_polytope_modal_t4_dispatch_20260514T134005Z__smoke__100ep_modal/harvested_artifacts/a1.bin \
  --output-dir experiments/results/d1_payload_budget_sweep_real_a1_20260515_codex \
  --policies rgb,green,neg_green,rb_pos_g_neg \
  --amplitude-scales 1.0 \
  --sign-policies payload,negate_payload \
  --payload-budget-bits 8000,20000,50000,100000 \
  --jacobian-lipschitz 20,10,5,2
```

Output manifest:

`experiments/results/d1_payload_budget_sweep_real_a1_20260515_codex/d1_overlay_policy_candidates_manifest.json`

Sweep summary:

- 128 byte-closed candidate packets materialized.
- All `8000` and `20000`-bit payloads remain decoded-zero under this allocator.
- First unblocked static-xray family: `budget=50000`, `L=2.0`.
- Smallest unblocked packet:
  - id: `d1_overlay_budget_50000_L_2_channel_rgb_amp_1_sign_payload`
  - archive bytes: `222129`
  - archive sha256:
    `1e027e2092e51c77c31185013d5fec81d67c283473ac1f85dc09a2421f33adc1`
  - decoded nonzero pixels: `4771 / 196608`
  - integer feasible pixels: `192714`
  - estimated changed bytes upper bound per pair: `74172`
- Larger-effect packet to test if tiny overlay underperforms:
  - id: `d1_overlay_budget_100000_L_2_channel_green_amp_1_sign_payload`
  - archive bytes: `223981`
  - archive sha256:
    `3c910bc7b26272087fc4c56d657d9506679aec22d8b283274983f0f18867ae4e`
  - decoded nonzero pixels: `130543 / 196608`
  - integer feasible pixels: `192714`
  - estimated changed bytes upper bound per pair: `674799`

No score claim. These are exact-eval candidates only after paired dispatch
claim + paired `[contest-CUDA]`/`[contest-CPU]` auth eval recovery.

## Nonzero D1 Paired Dispatch - 2026-05-15T17:49Z

Two nonzero payload-regenerated D1 candidates were dispatched as paired Modal
auth evals. Each pair uses the same archive/runtime on both axes.

| candidate | axis | call id | first recovery at 2026-05-15T17:50Z |
| --- | --- | --- | --- |
| `d1_b50k_L2_rgb_payload` | `[contest-CUDA]` | `fc-01KRPC42FCBHPYZ6M8WFDDKVQW` | pending |
| `d1_b50k_L2_rgb_payload` | `[contest-CPU]` | `fc-01KRPC4K5N0N1FK8D3PQGWQT51` | pending |
| `d1_b100k_L2_green_payload` | `[contest-CUDA]` | `fc-01KRPC42FWFAPZMXNDMNWVZFBX` | pending |
| `d1_b100k_L2_green_payload` | `[contest-CPU]` | `fc-01KRPC4K0RDQPR7JTFMFM7SC4B` | pending |

Dispatch plan artifacts:

- `experiments/results/d1_b50k_L2_rgb_payload_paired_auth_dispatch_plan_20260515T1755Z.json`
- `experiments/results/d1_b100k_L2_green_payload_paired_auth_dispatch_plan_20260515T1755Z.json`

Recovery commands:

```bash
.venv/bin/python tools/recover_modal_auth_eval.py \
  --output-dir experiments/results/modal_auth_eval/d1_b50k_L2_rgb_payload_paired_auth_20260515T1755Z_cuda

.venv/bin/python tools/recover_modal_auth_eval.py \
  --output-dir experiments/results/modal_auth_eval_cpu/d1_b50k_L2_rgb_payload_paired_auth_20260515T1755Z_cpu

.venv/bin/python tools/recover_modal_auth_eval.py \
  --output-dir experiments/results/modal_auth_eval/d1_b100k_L2_green_payload_paired_auth_20260515T1755Z_cuda

.venv/bin/python tools/recover_modal_auth_eval.py \
  --output-dir experiments/results/modal_auth_eval_cpu/d1_b100k_L2_green_payload_paired_auth_20260515T1755Z_cpu
```

No score claim yet; all four calls were pending on first recovery.

## Shrunk 96x128 Payload Sweep - 2026-05-15

The D1 materializer now supports area-downsampling the harvested real margin
map before payload regeneration via `--margin-map-resolution HxW`. This is the
intended D1 rate-axis correction: keep the overlay operational while reducing
the D1 sidecar from full-grid `384x512` cost to shrunk-grid `96x128` cost.

Command:

```bash
PYTHONPATH=src .venv/bin/python tools/build_d1_overlay_policy_candidates.py \
  --d1-bin experiments/results/lane_substrate_d1_segnet_margin_polytope_modal_t4_dispatch_20260514T134005Z__smoke__100ep_modal/harvested_artifacts/d1_polytope.bin \
  --a1-bin experiments/results/lane_substrate_d1_segnet_margin_polytope_modal_t4_dispatch_20260514T134005Z__smoke__100ep_modal/harvested_artifacts/a1.bin \
  --output-dir experiments/results/d1_payload_budget_sweep_real_a1_shrunk96_20260515_codex \
  --policies rgb,green,neg_green,rb_pos_g_neg \
  --amplitude-scales 1.0 \
  --sign-policies payload,negate_payload \
  --payload-budget-bits 8000,20000,50000,100000 \
  --jacobian-lipschitz 20,10,5,2 \
  --margin-map-resolution 96x128
```

Output manifest:

`experiments/results/d1_payload_budget_sweep_real_a1_shrunk96_20260515_codex/d1_overlay_policy_candidates_manifest.json`

Best static-xray rows:

| candidate | bytes | decoded nonzero | integer feasible | est changed bytes / pair | archive sha256 |
| --- | ---: | ---: | ---: | ---: | --- |
| `d1_overlay_budget_100000_L_2_res_96x128_channel_rgb_amp_1_sign_payload` | 185307 | 12288 | 12069 | 3052008 | `e8cd5499a1e5f012d60bb600ef8cf1d3b52ea54d11665bf76d4b18ec199c9aab` |
| `d1_overlay_budget_100000_L_2_res_96x128_channel_green_amp_1_sign_payload` | 185309 | 12288 | 12069 | 1017336 | `9195e42f80a55b84ed7e6950a96cdbeabfadb9a0f18b8d9bb9b320b096b648c0` |
| `d1_overlay_budget_50000_L_2_res_96x128_channel_rgb_amp_1_sign_payload` | 185310 | 12288 | 12069 | 3052008 | `b7dadbd2514fef90ae1963f4f48e5e4ccd6ec4ec675f9092bfc89175670e2b32` |

No score claim. The shrunk candidates dominate the full-grid D1 packets on
rate (`~185.3 KB` vs `~222.1 KB`) while preserving nonzero overlay diagnostics.

## Full-Grid Nonzero Paired Recovery - 2026-05-15T17:54Z

The first nonzero full-grid D1 packets scored cleanly on both axes, but both
measured configurations are worse than the A1/PR-family frontier. This is a
measured-config negative, not a D1 substrate death verdict.

| candidate | axis | archive bytes | canonical score | seg dist | pose dist | call id |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `d1_b50k_L2_rgb_payload` | `[contest-CUDA]` | 222129 | 0.2568473324645485 | 0.00067398 | 0.00017258 | `fc-01KRPC42FCBHPYZ6M8WFDDKVQW` |
| `d1_b50k_L2_rgb_payload` | `[contest-CPU]` | 222129 | 0.22352198595409703 | 0.00057337 | 0.00003341 | `fc-01KRPC4K5N0N1FK8D3PQGWQT51` |
| `d1_b100k_L2_green_payload` | `[contest-CUDA]` | 223981 | 0.26193283305735077 | 0.00070202 | 0.00018140 | `fc-01KRPC42FWFAPZMXNDMNWVZFBX` |
| `d1_b100k_L2_green_payload` | `[contest-CPU]` | 223981 | 0.2294549901682079 | 0.00060418 | 0.00003959 | `fc-01KRPC4K0RDQPR7JTFMFM7SC4B` |

Interpretation:

- Full-grid D1 pays too much rate and still increases both scorer components.
- CUDA is materially worse than CPU for the same D1 archives. Axis separation
  is mandatory for D1.
- Green-only is cheaper in changed-channel surface but the full-grid green
  payload was not enough to overcome its rate and distortion cost.

## Shrunk 96x128 Paired Dispatch And Recovery - 2026-05-15T18:00Z

Two shrunk candidates were dispatched paired because static xray showed a much
better rate surface than the full-grid packets.

| candidate | axis | archive bytes | canonical score | seg dist | pose dist | call id |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `d1_shrunk96_b100k_L2_rgb_payload` | `[contest-CUDA]` | 185307 | 0.24513262253913132 | 0.00073316 | 0.00023453 | `fc-01KRPCBT1PQKEH6Q5E490Z244T` |
| `d1_shrunk96_b100k_L2_rgb_payload` | `[contest-CPU]` | 185307 | 0.22071572261110622 | 0.00064128 | 0.00011022 | `fc-01KRPCCGRBKMZAP9ASN1VTK2C0` |
| `d1_shrunk96_b100k_L2_green_payload` | `[contest-CUDA]` | 185309 | 0.23714449393769646 | 0.00070756 | 0.00018489 | `fc-01KRPCC26M39FN0SWPCZJZMX35` |
| `d1_shrunk96_b100k_L2_green_payload` | `[contest-CPU]` | 185309 | 0.20736549114794994 | 0.00061258 | 0.00005161 | `fc-01KRPCCM97MT1RAW6NN2HPX9XV` |

Interpretation:

- Shrinking the D1 sidecar fixed the rate-axis waste but not the scorer
  objective mismatch.
- Green-only is consistently less bad than RGB on both axes.
- Dense `100000`-bit D1 is still too aggressive: it raises SegNet and PoseNet
  error more than the rate reduction can recover.

## Sparse 96x128 Refinement - 2026-05-15T18:03Z

Allocator inspection showed a missed sparse region: `budget=5000` at `96x128`
modifies `4470 / 12288` decoded lattice sites versus `12288 / 12288` for the
dense `100000` packet. This is a real D1 frontier correction: optimize the
distortion surface, not just archive bytes.

Command:

```bash
PYTHONPATH=src .venv/bin/python tools/build_d1_overlay_policy_candidates.py \
  --d1-bin experiments/results/lane_substrate_d1_segnet_margin_polytope_modal_t4_dispatch_20260514T134005Z__smoke__100ep_modal/harvested_artifacts/d1_polytope.bin \
  --a1-bin experiments/results/lane_substrate_d1_segnet_margin_polytope_modal_t4_dispatch_20260514T134005Z__smoke__100ep_modal/harvested_artifacts/a1.bin \
  --output-dir experiments/results/d1_sparse_shrunk96_refine_real_a1_20260515_codex \
  --policies green,neg_green \
  --amplitude-scales 0.5,1.0 \
  --sign-policies payload,negate_payload \
  --payload-budget-bits 5000,8000,12000,16000 \
  --jacobian-lipschitz 2 \
  --margin-map-resolution 96x128
```

Smallest sparse score-bearing candidates:

| candidate | bytes | decoded nonzero | est changed bytes / pair | archive sha256 |
| --- | ---: | ---: | ---: | --- |
| `d1_overlay_budget_5000_L_2_res_96x128_channel_green_amp_1_sign_payload` | 185830 | 4470 | 369444 | `bcabbb509b225e68b24346743921a4c15d5073ec60df98b5fd339f6e02dbb291` |
| `d1_overlay_budget_5000_L_2_res_96x128_channel_green_amp_1_sign_negate_payload` | 185834 | 4470 | 369444 | `adb4566c5db1eecf180843a9828a3cd7e24b04a656041762a879e8ec11588cfe` |

Paired dispatch:

| candidate | axis | call id | recovery status |
| --- | --- | --- | --- |
| `d1_sparse96_b5k_L2_green_payload` | `[contest-CUDA]` | `fc-01KRPCQ0HXBQFSEMFJ8ZEAEZBW` | recovered: 0.23354941003137458 |
| `d1_sparse96_b5k_L2_green_payload` | `[contest-CPU]` | `fc-01KRPCQJ04MX9YFTTJD7M9X97H` | recovered: 0.2011587759556391 |
| `d1_sparse96_b5k_L2_green_negate` | `[contest-CUDA]` | `fc-01KRPCQ0KMZ1F14TAND8WEC16D` | recovered: 0.2342490775283781 |
| `d1_sparse96_b5k_L2_green_negate` | `[contest-CPU]` | `fc-01KRPCQHV73A4CT26TVG07XE0F` | recovered: 0.20143815804513954 |

The sparse sign test improves CUDA over the dense shrunk green packet by about
`0.0036` and CPU by about `0.0062`, but still remains above the sub-0.192
submission gate. D1 must move to scorer-aware sign/mask selection instead of
static margin-only lattice placement.

## Sparse Alternating-Pair Sign Dispatch - 2026-05-15T18:05Z

One final cheap static D1 hypothesis was dispatched: keep the sparse `5000`
green lattice but alternate sign by pair to reduce systematic PoseNet drift.

| candidate | axis | call id | status |
| --- | --- | --- | --- |
| `d1_sparse96_b5k_L2_green_alternating` | `[contest-CUDA]` | `fc-01KRPD0K6P1SGCF5AVSGFCY7YH` | recovered: 0.2339322883249223 |
| `d1_sparse96_b5k_L2_green_alternating` | `[contest-CPU]` | `fc-01KRPD13T901AD636FJ58RZ39E` | recovered: 0.2011832266102615 |

Alternating did not beat same-sign sparse on either axis. Best measured static
D1 remains:

- `[contest-CUDA]`: `d1_sparse96_b5k_L2_green_payload` at
  `0.23354941003137458`.
- `[contest-CPU]`: `d1_sparse96_b5k_L2_green_payload` at
  `0.2011587759556391`.

Both are valid exact measurements and valid negatives for the static D1 overlay
family, but they are not promotion candidates and they remain above the
sub-0.192 submission gate.

## D1 Safety And Manifest Hardening - 2026-05-15T18:09Z

Fresh-eyes adversarial review found that the prior D1 encoder allocated
integer lattice values from entropy buckets without clamping the actual
magnitude to `floor(margin/L)`. That means exact-eval numbers above remain
legitimate measurements, but the older method claim "inside the SegNet
polytope" was under-certified.

Fixes landed:

- `allocate_noise_within_polytope(...)` now clamps every lattice value to the
  per-pixel integer safe budget `floor(margin/L)`.
- `validate_polytope_margin_contract(...)` and the generated D1 runtime
  `_decode_overlay(...)` now fail closed if decoded noise exceeds
  `floor(margin/L)`.
- `analyze_d1_overlay_effect(...)` now reports `unsafe_nonzero_pixels` and
  blocks exact-eval dispatch when unsafe lattice entries are present.
- Full trainer shrunk margin extraction now explicitly uses `downsample_mode="area"`
  whenever target resolution differs from the canonical scorer plane, avoiding
  silent bilinear/area drift between trainer-produced and materializer-produced
  shrunk D1 sidecars.

New diagnostics added after the sparse finding:

- `decoded_noise_abs_sum`
- `camera_overlay_abs_sum`
- `attenuated_overlay_abs_sum`
- `estimated_changed_lsb_l1_upper_bound_per_pair`
- `estimated_changed_lsb_l2_energy_upper_bound_per_pair`
- `unsafe_nonzero_pixels`

The previous diagnostics counted changed pixels but not magnitude; that made
`+1` and `+2` overlays look too similar. The builder now also emits
`overlay_effect_equivalence_key` and `duplicate_of_candidate_id`, marking
metadata-different packets that render identical signed deltas, for example
`green + negate_payload` versus `neg_green + payload`.

Verification:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  src/tac/tests/test_build_d1_overlay_policy_candidates.py \
  src/tac/tests/test_xray_d1_overlay_payload.py \
  src/tac/substrates/d1_segnet_margin_polytope/tests

ruff check \
  src/tac/substrates/d1_segnet_margin_polytope/diagnostics.py \
  src/tac/substrates/d1_segnet_margin_polytope/overlay.py \
  src/tac/substrates/d1_segnet_margin_polytope/polytope_encoder.py \
  experiments/train_substrate_d1_segnet_margin_polytope.py \
  tools/build_d1_overlay_policy_candidates.py \
  src/tac/substrates/d1_segnet_margin_polytope/tests/test_d1_overlay_and_shrink.py \
  src/tac/substrates/d1_segnet_margin_polytope/tests/test_d1_substrate.py \
  src/tac/tests/test_build_d1_overlay_policy_candidates.py
```

Results: `125 passed`; ruff clean; `git diff --check` clean.

Next D1 action is a post-fix *certified* sparse96 kink sweep below 5k:
`budget_bits=500,1000,1500,2000,3000,4000,5000`,
`channel={green,blue,red}`, `sign={payload,negate_payload,alternating_pairs}`.
Only rows with `unsafe_nonzero_pixels=0`, nonzero xray, and lower changed-LSB
surface than the measured 5k packets should be paired-dispatched.

## Post-Fix Certified b3k Paired Recovery - 2026-05-15T18:23Z

Recovered exact paired Modal auth eval for certified sparse96 3k packets:

| candidate | axis | call id | bytes | score | pose | seg |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `d1_cert_sparse96_b3k_L2_green_payload` | `[contest-CUDA]` | `fc-01KRPDEXYQKBC4WSFKVV8ZY7G6` | 185379 | 0.2317362548377786 | 0.00017128 | 0.00066914 |
| `d1_cert_sparse96_b3k_L2_blue_payload` | `[contest-CUDA]` | `fc-01KRPDF3G9H1HCBAASJF5BY2SG` | 185378 | 0.231727964404999 | 0.00017125 | 0.00066910 |
| `d1_cert_sparse96_b3k_L2_red_payload` | `[contest-CUDA]` | `fc-01KRPDEZR37GGGC6CRSEVAVEC2` | 185376 | 0.23175416774323881 | 0.00017142 | 0.00066917 |
| `d1_cert_sparse96_b3k_L2_green_payload` | `[contest-CPU]` | `fc-01KRPDFEZ9QMPA4G3FDFACRV63` | 185379 | 0.19836541637745966 | 0.00003299 | 0.00056766 |
| `d1_cert_sparse96_b3k_L2_blue_payload` | `[contest-CPU]` | `fc-01KRPDFKZHX8C0C8M05JHJ8XTT` | 185378 | 0.1982817145388956 | 0.00003291 | 0.00056705 |
| `d1_cert_sparse96_b3k_L2_red_payload` | `[contest-CPU]` | `fc-01KRPDFG7E61AGF0CZYG5B64A2` | 185376 | 0.19839589432454263 | 0.00003293 | 0.00056815 |

Interpretation:

- b3k improves over the prior b5k CPU exact row (`0.2011587759556391` ->
  `0.1982817145388956` best CPU), but remains above the sub-0.192 gate.
- b3k CUDA remains around `0.23173`, so static global D1 overlay is not a
  promotion path.
- The valid next D1 optimization is not another static same-sign sweep; it is
  a scorer-guided selector that applies the overlay only where pair-level xray
  says it helps.

## Pair-Mask Selector Wiring - 2026-05-15T18:58Z

Implemented a byte-closed D1 `pair_mask` sign policy:

- `pack_pair_sign_mask(...)` / `unpack_pair_sign_mask(...)` encode `-1/0/+1`
  per pair in a deterministic 2-bit stream.
- Runtime-generated `inflate.py` decodes the compact pair-mask metadata,
  validates exact `n_pairs`, skips disabled pairs, and applies positive or
  negated overlays per selected pair. The current canonical metadata surface is
  `pair_mask_b85` + `pair_mask_n`; the earlier hex/base64 surfaces are
  superseded below.
- `tools/build_d1_pair_mask_from_xray.py` converts pair-component xray JSON
  into a selector packet. The selector is explicitly `score_claim=false`.
- `tools/build_d1_overlay_policy_candidates.py` now materializes pair-mask
  packets and records mask SHA, active pair counts, and dispatch blockers.

Important math hardening:

The first selector draft minimized per-pair
`sqrt(10 * pose_dist) + 100 * seg_dist`. That is not the contest objective,
which is `sqrt(10 * mean_pose_dist) + 100 * mean_seg_dist`. The selector now
uses `contest_score_linearized_at_baseline_mean_pose_v1`:

```text
pose_weight = 5 / sqrt(10 * baseline_mean_pose_dist)
seg_weight  = 100
pair objective = pose_weight * pose_dist + seg_weight * seg_dist
```

This prevents a repeated objective-mismatch failure class where pairwise
`sqrt(pose)` improvements look larger than their actual global-score effect.

Local CPU xray evidence:

| packet | pairs | pose | seg | component no-rate |
| --- | ---: | ---: | ---: | ---: |
| A1 baseline local CPU xray | 600 | 0.00003286083487049988 | 0.0005604044600234677 | 0.07416800371719073 |
| D1 global green +payload local CPU xray | 600 | 0.000032985370828176505 | 0.0005676269537070766 | 0.07492457050192043 |
| D1 positive-only pair mask local CPU xray | 600 | 0.00003272931126995597 | 0.0005596160889156939 | 0.07405285298089502 |
| D1 `+/-/off` selector, first 64-pair runtime smoke | 64 | 0.000020995494973696793 | 0.0005544821442526882 | 0.06993803670633797 |

The full 600-pair `+/-/off` selector from positive and negative xray chose
235 pairs (`126` positive, `109` negative) and predicts
`predicted_component_no_rate_delta=-0.00019934154650622093` before byte cost.
Its materialized archive is:

```text
experiments/results/d1_pair_mask_cpu600_posneg_candidate_20260515_codex/d1_overlay_channel_green_amp_1_sign_pair_mask_pairmask_cpu600p235pn/archive.zip
bytes=185700
sha256=32443f4dbbcd41b4a3b7f7e86160e3b0a7f0e07541da4974aa0dded2a53cc4fd
```

The selector is technically valid and useful for future per-pair action masks,
but it does **not** justify paired paid dispatch by itself: local predicted
CPU score remains about `0.1976`, and the measured static CUDA axis is far
above the gate. The next D1 frontier move, if reopened, should be a
per-pair action selector over channel/sign/amplitude or a CUDA-native xray
selector; single-channel `green + +/-/off` is not enough.

## Pair-Mask Rate-Aware Waterfill Guard - 2026-05-15

`tools/build_d1_pair_mask_from_xray.py` now chooses pair-mask actions with a
rate-aware prefix search instead of blindly enabling every positive linearized
pair. It still ranks candidate per-pair signs by the linearized contest
objective, but then evaluates the actual global
`sqrt(10 * mean_pose_dist) + 100 * mean_seg_dist` for each prefix and subtracts
the fixed pair-mask byte cost through the contest rate term. If no nonzero mask
pays for its bytes, it emits an all-zero selector and records the skipped
component-only prefix for audit.

Regression guard:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  src/tac/tests/test_build_d1_pair_mask_from_xray.py
# 2 passed
```

Applied to the existing CPU600 positive/negative D1 xray with the observed
pair-mask overhead (`185700 - 185379 = 321` bytes):

```bash
.venv/bin/python tools/build_d1_pair_mask_from_xray.py \
  --baseline-xray experiments/results/d1_pair_component_xray_a1_baseline_cpu_20260515_codex/pair_component_xray.json \
  --positive-xray experiments/results/d1_pair_component_xray_cert_sparse96_b3k_green_cpu_20260515_codex/pair_component_xray.json \
  --negative-xray experiments/results/d1_pair_component_xray_negative_green_b3k_20260515_codex/pair_component_xray.json \
  --evidence-axis local_cpu_xray \
  --incremental-rate-cost-bytes 321 \
  --incremental-baseline-label d1_static_b3k_green_packet \
  --output-n-pairs 600 \
  --output-json experiments/results/d1_pair_mask_selector_cpu600_posneg_rateaware_20260515_codex/pair_mask_600.json
```

Result: `potential_pairs=235`, `best_component_prefix_size=235`, but
`active_pairs=0` after rate. The prior 235-pair mask had
`predicted_component_no_rate_delta=-0.00019934154650622093`; the fixed 321-byte
mask costs `0.000213740723952217` score, so the rate-aware selector correctly
blocks the packet. This prevents another paid paired eval of a candidate whose
own local model says it cannot pay for its selector bytes.

## Pair-Mask Metadata Compression - 2026-05-15

The rate-aware selector exposed a narrow but real engineering gap under a
same-family incremental accounting model: the 235-pair CPU600 `+/-/off` mask
missed break-even by only about 22 archive bytes. This was first tightened by
moving from hex metadata to base64-encoded 2-bit bytes, and was later
superseded by the base85 minimum-metadata packet below. For a 600-pair selector
the base64 intermediate changed the metadata payload from 300 JSON characters
to 200 JSON characters while keeping the same 150 raw selector bytes and the
same `[-1, 0, +1]` semantics.

Code surfaces:

- `src/tac/substrates/d1_segnet_margin_polytope/overlay.py`
- `src/tac/substrates/d1_segnet_margin_polytope/inflate.py`
- `experiments/train_substrate_d1_segnet_margin_polytope.py`
- `tools/build_d1_overlay_policy_candidates.py`

Regression guards:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  src/tac/substrates/d1_segnet_margin_polytope/tests/test_d1_overlay_and_shrink.py \
  src/tac/substrates/d1_segnet_margin_polytope/tests/test_d1_substrate.py \
  src/tac/tests/test_build_d1_overlay_policy_candidates.py \
  src/tac/tests/test_build_d1_pair_mask_from_xray.py
# 119 passed
```

```bash
.venv/bin/ruff check \
  src/tac/substrates/d1_segnet_margin_polytope/overlay.py \
  src/tac/substrates/d1_segnet_margin_polytope/inflate.py \
  experiments/train_substrate_d1_segnet_margin_polytope.py \
  tools/build_d1_overlay_policy_candidates.py \
  src/tac/substrates/d1_segnet_margin_polytope/tests/test_d1_overlay_and_shrink.py \
  src/tac/substrates/d1_segnet_margin_polytope/tests/test_d1_substrate.py \
  src/tac/tests/test_build_d1_overlay_policy_candidates.py
# All checks passed
```

Materialized byte-closed candidate:

```text
experiments/results/d1_pair_mask_cpu600_posneg_b64_rateaware_candidate_20260515_codex/d1_overlay_channel_green_amp_1_sign_pair_mask_pairmask_cpu600p235pn_b64_rateaware/archive.zip
bytes=185673
sha256=6dae91c151b083b391ec88fd045dc86f5c89fb783c0cbf74d27c25b780a3065f
```

This is 27 bytes smaller than the previous hex pair-mask packet (`185700`).
Using the same CPU600 xray component estimate:

```text
component delta no-rate = -0.00019934154650622093
old selector rate bytes = 321 -> net +0.000014399177445996057
new selector rate bytes = 294 -> net -0.000003579014288302557
```

This was an intermediate same-family incremental estimate only. It is
CPU-xray-derived and **not** a promotion or submission result. Full A1-relative
archive accounting is recorded in the supersession block below and blocks the
pair-mask packet.

## Pair-Mask Minimum Metadata Tightening - 2026-05-15T19:34Z

Further D1 review found one more score-bearing selector leak: the archive
metadata stored `overlay_pair_sign_mask_sha256` inside `d1_polytope.bin`.
That hash is useful custody signal, but inflate does not need it; keeping it
inside the scored packet spends bytes. D1 now keeps the selector hash in the
candidate manifest only, and the scored D1 metadata carries only:

```text
pair_mask_b85=<base85 2-bit selector bytes>
pair_mask_n=<pair count>
```

The runtime and reusable overlay decoder consume `pair_mask_b85` directly.
The old long metadata keys are removed from newly materialized packets:
`overlay_pair_sign_mask_b64`, `overlay_pair_sign_mask_bits_hex`, and
`overlay_pair_sign_mask_sha256`.

Regression hardening:

- `analyze_d1_overlay_effect(...)` now reports `pair_mask_active_pairs`.
- All-zero pair-mask packets now get the explicit blocker
  `d1_pair_mask_has_no_active_pairs`, preventing a repeat dead-rate dispatch
  where static overlay diagnostics are nonzero but every pair is disabled.

Verification:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  src/tac/substrates/d1_segnet_margin_polytope/tests/test_d1_overlay_and_shrink.py \
  src/tac/substrates/d1_segnet_margin_polytope/tests/test_d1_substrate.py \
  src/tac/tests/test_build_d1_overlay_policy_candidates.py \
  src/tac/tests/test_build_d1_pair_mask_from_xray.py \
  src/tac/tests/test_xray_d1_overlay_payload.py
# 121 passed

.venv/bin/ruff check \
  src/tac/substrates/d1_segnet_margin_polytope/overlay.py \
  src/tac/substrates/d1_segnet_margin_polytope/diagnostics.py \
  experiments/train_substrate_d1_segnet_margin_polytope.py \
  tools/build_d1_overlay_policy_candidates.py \
  src/tac/substrates/d1_segnet_margin_polytope/tests/test_d1_overlay_and_shrink.py \
  src/tac/substrates/d1_segnet_margin_polytope/tests/test_d1_substrate.py \
  src/tac/tests/test_build_d1_overlay_policy_candidates.py
# All checks passed
```

Materialized byte-closed candidate:

```text
experiments/results/d1_pair_mask_cpu600_posneg_b85_minmeta_candidate_20260515_codex/d1_overlay_channel_green_amp_1_sign_pair_mask_pairmask_cpu600p235pn_b85_minmeta/archive.zip
bytes=185593
sha256=d3f21f3512d61d6d5d4f7625cb38d54fac9f389f2ef7af766e64622d62d0848e
pair_mask_active_pairs=235
```

Byte delta versus previous D1 pair-mask packets:

```text
hex metadata packet: 185700 bytes
base64 packet:       185673 bytes
base85 min-meta:     185593 bytes
```

Using the same CPU600 xray component estimate against the static D1 packet
family:

```text
component delta no-rate = -0.00019934154650622093
old b64 selector rate bytes = 294 -> net -0.000003579014288302557
new min-meta rate bytes = 214 -> net -0.00005684523562047961
```

Important correction: this incremental accounting is valid only inside the D1
static-packet family. It is **not** valid for an A1-relative promotion claim,
because the xray baseline rows were A1. Full A1-relative rate accounting must
charge the whole D1 packet delta.

Full-rate selector audit:

```bash
.venv/bin/python tools/build_d1_pair_mask_from_xray.py \
  --baseline-xray experiments/results/d1_pair_component_xray_a1_baseline_cpu_20260515_codex/pair_component_xray.json \
  --positive-xray experiments/results/d1_pair_component_xray_cert_sparse96_b3k_green_cpu_20260515_codex/pair_component_xray.json \
  --negative-xray experiments/results/d1_pair_component_xray_negative_green_b3k_20260515_codex/pair_component_xray.json \
  --evidence-axis local_cpu_xray \
  --baseline-archive-bytes 178162 \
  --candidate-archive-bytes 185593 \
  --output-n-pairs 600 \
  --output-json experiments/results/d1_pair_mask_selector_cpu600_posneg_fullrate_20260515_codex/pair_mask_600.json
```

Result:

```text
potential_pairs=235
best_component_prefix_size=235
best_component_no_rate_delta=-0.00019934154650622093
archive_byte_delta=7431
rate_penalty_score=0.004947997880650855
active_pairs=0
predicted_score_lowering_after_rate=false
```

`tools/build_d1_pair_mask_from_xray.py` now requires an explicit
`--evidence-axis` plus exactly one rate scope:
`--baseline-archive-bytes/--candidate-archive-bytes` for full A/B accounting,
or `--incremental-rate-cost-bytes/--incremental-baseline-label` for
same-family incremental selector accounting. This permanently prevents the
specific false-positive class where an A1-relative xray selector charges only
incremental D1 metadata bytes.

This is still not a submission candidate and does not fix D1's CUDA-axis
weakness. It is a real byte-closed D1 packet-size improvement and a stronger
selector guard; the next meaningful D1 move remains CUDA-native pair/action
xray or a per-pair action selector over channel/sign/amplitude, not another
static same-sign sweep.

## Pair-Mask Adversarial Guardrail Supersession - 2026-05-15T19:48Z

Recursive adversarial review found two remaining false-positive paths:

1. `tools/build_d1_pair_mask_from_xray.py` trusted a caller-supplied axis string
   instead of verifying xray provenance.
2. Full-rate accounting still allowed hand-typed archive bytes, which could
   reintroduce the exact A1-relative-vs-incremental selector mistake.

Hardening landed:

- Xray reports now must expose verifiable provenance (`schema`,
  `evidence_grade`/`device`, row count, file SHA), and `--evidence-axis` must
  match all input xray axes. The current CPU xray artifacts are therefore
  fixed as `local_cpu_xray`; they cannot be relabeled as `contest_cuda`.
- D1 policy candidate manifests now record `base_member_bytes`,
  `source_base_archive_bytes`, `archive_delta_vs_source_base_archive_bytes`,
  `d1_sidecar_bytes`, and the relevant SHA-256 custody fields.
- The pair-mask selector can read full-rate bytes directly from
  `--rate-from-candidate-manifest`, avoiding manual byte transcription.
- Pair-mask materialization defaults to `--expected-pairs 600`; partial masks
  require `--allow-partial-smoke` and receive a non-contest dispatch blocker.
- Negative archive deltas require an explicit waiver and rationale.

Manifest-sourced full-rate audit:

```bash
PYTHONPATH=src .venv/bin/python tools/build_d1_pair_mask_from_xray.py \
  --baseline-xray experiments/results/d1_pair_component_xray_a1_baseline_cpu_20260515_codex/pair_component_xray.json \
  --positive-xray experiments/results/d1_pair_component_xray_cert_sparse96_b3k_green_cpu_20260515_codex/pair_component_xray.json \
  --negative-xray experiments/results/d1_pair_component_xray_negative_green_b3k_20260515_codex/pair_component_xray.json \
  --evidence-axis local_cpu_xray \
  --rate-from-candidate-manifest experiments/results/d1_pair_mask_cpu600_posneg_b85_minmeta_candidate_20260515_codex/d1_overlay_channel_green_amp_1_sign_pair_mask_pairmask_cpu600p235pn_b85_minmeta/candidate_manifest.json \
  --output-json experiments/results/d1_pair_mask_selector_cpu600_posneg_manifest_fullrate_20260515_codex/pair_mask_600.json
```

Result:

```text
archive_byte_delta=7431
rate_penalty_score=0.004947997880650855
best_component_no_rate_delta=-0.00019934154650622093
active_pairs=0
predicted_score_lowering_after_rate=false
```

Verification:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  src/tac/substrates/d1_segnet_margin_polytope/tests/test_d1_overlay_and_shrink.py \
  src/tac/substrates/d1_segnet_margin_polytope/tests/test_d1_substrate.py \
  src/tac/tests/test_build_d1_overlay_policy_candidates.py \
  src/tac/tests/test_build_d1_pair_mask_from_xray.py \
  src/tac/tests/test_xray_d1_overlay_payload.py
# 127 passed
```

Conclusion: D1 pair-mask min-meta is a legitimate byte-closed engineering
improvement and a useful selector-guard artifact, but the full A1-relative,
manifest-sourced rate audit blocks it. It remains non-promotional and should
not be submitted; useful D1 work now needs CUDA-native pair/action xray or a
new action family with enough component movement to pay the full sidecar cost.

## Xray And Dispatch-Protocol Hardening Addendum - 2026-05-15T22:20Z

Additional D1 hardening landed after the adversarial review:

- `tools/xray_d1_overlay_payload.py` now decodes pair-mask archives from
  metadata and also supports an explicit `--pair-sign-mask-json` override for
  selector debugging. Pair-mask xray reports now name the mask source instead
  of silently analyzing a zero-action or wrong-action payload.
- `build_readiness_manifest(...)` now consumes `unsafe_nonzero_pixels`,
  `pair_mask_active_pairs`, and overlay dispatch blockers. A D1 packet with
  unsafe pixels, zero active pair-mask pairs, or unresolved overlay blockers is
  `ready_for_exact_eval_dispatch=false`.
- Both D1 and DP1 recipes now declare the full hardware feasibility contract
  (`min_vram_gb` plus `min_smoke_gpu`), so the new Boyd-style
  `dispatch_protocol_complete` umbrella can reason about the conjunction of
  engineering, hardware, and substrate correctness before any paid dispatch.

Verification:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  src/tac/substrates/d1_segnet_margin_polytope/tests \
  src/tac/tests/test_build_d1_overlay_policy_candidates.py \
  src/tac/tests/test_build_d1_pair_mask_from_xray.py \
  src/tac/tests/test_xray_d1_overlay_payload.py
# 142 passed

PYTHONPATH=src .venv/bin/python tools/check_dispatch_protocol_complete.py \
  --recipe .omx/operator_authorize_recipes/substrate_d1_segnet_margin_polytope_modal_t4_dispatch.yaml
# dispatch_protocol_complete=false; tier1 blocks on dispatch_enabled=false
# and the explicit D1 L1/L2 overlay blockers; tier2/tier3 pass.
```

Status remains unchanged: D1 is still non-promotional. The correct next D1
frontier step is not another static film-grain sweep; it is CUDA-native
pair/action xray with a rate-aware selector over channel, sign, amplitude, and
pair subset, followed by paired CPU/CUDA exact eval only if the readiness
manifest turns green.
