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
