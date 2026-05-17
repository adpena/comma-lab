# L5 v2 TT5L current-code nonzero side-info advisory smoke

Date: 2026-05-17
Author: Codex
Scope: L5 v2 Time-Traveler staircase, side-info liveness and effect-curve custody
Evidence grade: `[macOS-CPU advisory]`

## Summary

Current-code TT5L training can now emit nonzero side-info, unlike the old 25ep
Modal archive classified in
`l5_v2_tt5l_all_zero_sideinfo_failure_classification_20260517_codex.md`.
This is not a score claim, not a contest-shape run, and not promotion-eligible:
the smoke used 2/600 pairs, CPU device, `--skip-auth-eval`, and the advisory
waiver intended only to test archive liveness.

## Command

```bash
OUT="experiments/results/time_traveler_l5_v2/tt5l_current_code_nonzero_sideinfo_cpu_advisory_20260517T025048Z"
.venv/bin/python experiments/train_substrate_time_traveler_l5_autonomy.py \
  --output-dir "$OUT" \
  --epochs 1 \
  --device cpu \
  --full-cpu \
  --advisory-cpu-explicitly-waived \
  --max-pairs 2 \
  --batch-size 1 \
  --val-pair-count 1 \
  --val-every-epochs 1 \
  --skip-auth-eval \
  --max-wall-clock-hours 0.25
```

## Artifact Custody

- Output dir: `experiments/results/time_traveler_l5_v2/tt5l_current_code_nonzero_sideinfo_cpu_advisory_20260517T025048Z`
- Archive: `experiments/results/time_traveler_l5_v2/tt5l_current_code_nonzero_sideinfo_cpu_advisory_20260517T025048Z/archive.zip`
- Archive bytes: `27147`
- Archive SHA-256: `33f27f82649b08af0bb1ea987911a96f943a397603f42eab8d3cca83d700d6b4`
- `0.bin` bytes: `27029`
- `0.bin` SHA-256: `1e5166bb6273a51d664d3bcec89b319f461d9213a3ad025cbffbe631a9866e82`
- Best validation lag: `93.50102233886719` at epoch `0`
- Device: `cpu`
- Auth eval: skipped

## Side-Info Liveness

The archive parser reports side-info shape `[2, 45]` with `62/90` nonzero
values. Both pairs carry nonzero side-info (`2/2` nonzero pairs, `0/2`
all-zero pairs), with `31` nonzero values per pair.

Per-section liveness:

| Section | Nonzero | Total | Fraction |
| --- | ---: | ---: | ---: |
| `se3_lie` | 17 | 24 | 0.7083 |
| `seg_boundary` | 25 | 36 | 0.6944 |
| `hf_residual` | 7 | 12 | 0.5833 |
| `predict_residual` | 13 | 18 | 0.7222 |

This closes only the "current code can emit nonzero side-info" question. It
does not close the L5 v2 side-info effect curve because the run is not full
contest shape and has no paired CPU/CUDA exact eval.

## Variant Packet Manifest

Built byte-closed TT5L side-info control packets from this advisory source:

- JSON: `.omx/research/l5_v2_tt5l_current_code_nonzero_sideinfo_variant_packets_20260517_codex.json`
- Markdown: `.omx/research/l5_v2_tt5l_current_code_nonzero_sideinfo_variant_packets_20260517_codex.md`
- Variants: `zero`, `random_lsb`, `shuffled`, `trained`, `ablated`

Top-level blockers:

- `requires_paired_cpu_cuda_exact_eval_for_sideinfo_effect_curve`
- `requires_dispatch_lane_claim_before_auth_eval`
- `score_claim_forbidden_until_effect_curve_artifact_passes`
- `tt5l_source_num_pairs_not_full_contest:2_expected_600`

The final blocker was added in this landing so partial-pair timing/advisory
archives cannot look dispatch-ready or effect-curve-complete by omission.

## Classification

Verdict: `advisory_liveness_green_promotion_blocked`

The current-code L5 v2 TT5L packet has live side-info bytes and can support the
next effect-curve build, but this measured artifact is explicitly non-promotional
until a claimed full-shape run produces paired `[contest-CPU]` and
`[contest-CUDA]` exact-eval cells for the required side-info variants.

Next action: run a claimed provider/GHA Linux CPU plus CUDA full 600-pair TT5L
current-code timing or short run, rebuild the five side-info variants from that
source archive, then feed the paired cells into the L5 v2 effect-curve and
dispatch-plan artifacts.
