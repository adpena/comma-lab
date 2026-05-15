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

