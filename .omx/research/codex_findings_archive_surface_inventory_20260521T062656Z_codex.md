# Codex Findings: Archive Surface Inventory

- timestamp_utc: 2026-05-21T06:26:56Z
- lane: archive_surface_inventory_after_atw2_candidate_blocker
- status: LANDED_DIAGNOSTIC_TOOL_PLUS_FULL_LOCAL_SCAN
- score_claim: false
- promotion_eligible: false
- ready_for_exact_eval_dispatch: false

## Why this exists

ATW2 CDF compaction is locally blocked on missing full candidate archives: the
local CPU trainer refuses non-smoke generation and this machine has no CUDA
runtime. Instead of adding another speculative plan, I built a local archive
surface scanner to identify existing `archive.zip` byte surfaces that can be
ranked before another lane-specific actuator is written.

Tool landed:

- `tools/build_archive_surface_inventory.py`

The tool scans `archive.zip` files, records ZIP member structure, duplicate
archive SHA clusters, and a simple member-entropy headroom estimate. It is a
diagnostic chooser only. It does not claim that the estimated headroom is
reachable, scorer-neutral, or promotion-safe.

## Verification commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/build_archive_surface_inventory.py \
  experiments/results submissions \
  --max-archives 5 \
  --output-dir experiments/results/archive_surface_inventory_smoke_20260521T0635Z \
  --top 5 >/tmp/archive_surface_inventory_smoke.json

git diff --check -- tools/build_archive_surface_inventory.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file \
  tools/build_archive_surface_inventory.py --status reviewed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check \
  tools/build_archive_surface_inventory.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/build_archive_surface_inventory.py \
  experiments/results submissions \
  --output-dir experiments/results/archive_surface_inventory_20260521T062618Z \
  --top 25 > experiments/results/archive_surface_inventory_20260521T062618Z/stdout.json
```

Review tracker result:

- `tools/build_archive_surface_inventory.py`: NORMAL
- 16 entities reviewed
- 0 policy violations

## Full inventory artifacts

- JSON: `experiments/results/archive_surface_inventory_20260521T062618Z/archive_surface_inventory.json`
- Markdown: `experiments/results/archive_surface_inventory_20260521T062618Z/archive_surface_inventory.md`
- stdout mirror: `experiments/results/archive_surface_inventory_20260521T062618Z/stdout.json`

SHA-256:

```text
3745c9a0da8b31d3cd3a1ca2ca09f9e2cf1ee12d8466233bfdefd6968911908a  archive_surface_inventory.json
7736638450fc8ca901cac35ab03eafe17c42687c0e441821bfca89da2144102c  archive_surface_inventory.md
3745c9a0da8b31d3cd3a1ca2ca09f9e2cf1ee12d8466233bfdefd6968911908a  stdout.json
```

Byte counts:

```text
3536088  archive_surface_inventory.json
  22867  archive_surface_inventory.md
3536088  stdout.json
```

## Full-scan results

- Archives seen: 3,786
- ZIP errors: 1
- ZIP error path:
  `experiments/results/top_submission_delta_reverse_engineering_20260503/pr_heads/pr65/submissions/henosis_qz_n3z_r25_clean/archive.zip`
- Shape counts:
  - `multi_member_archive`: 1,532
  - `single_unknown_member`: 1,174
  - `single_x_member`: 701
  - `single_0bin_member`: 354
  - `canonical_renderer_masks_pose_members`: 24

Top estimated recoverable ZIP surfaces by member entropy floor:

| rank | archive | archive bytes | recoverable bytes | rate delta if floor reached | shape |
|---:|---|---:|---:|---:|---|
| 1 | `experiments/results/theoretical_floor_lfv1_pose_foveation_adapter_20260520_codex/archive_candidate/archive.zip` | 131418 | 60799 | -0.0404835584908749 | `multi_member_archive` |
| 2 | `experiments/results/t1_codex_smoke_20260509/archive.zip` | 560394 | 42418 | -0.028244405073536264 | `single_x_member` |
| 3 | `experiments/results/t1_phase1_scaffold_smoke/archive.zip` | 560394 | 42370 | -0.0282124438437864 | `single_x_member` |
| 4 | `experiments/results/theoretical_floor_lfv1_pose_foveation_bridge_20260520_codex/archive_candidate/archive.zip` | 84971 | 40901 | -0.02723429704164993 | `multi_member_archive` |
| 5 | `experiments/results/theoretical_floor_lfv1_pose_foveation_20260520_codex/archive_candidate/archive.zip` | 68460 | 34312 | -0.022846952399527942 | `multi_member_archive` |
| 6 | `experiments/results/lapose_foveation_payload_candidate_20260513_codex/archive_candidate/archive.zip` | 68423 | 34306 | -0.02284295724580921 | `multi_member_archive` |
| 7 | `experiments/results/hfv1_sidecar_candidates_20260520_codex/pose_top8_alpha00045_radius070/archive/archive.zip` | 202649 | 14722 | -0.009802775507864606 | `multi_member_archive` |
| 8 | `experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/archive_identity/archive.zip` | 202649 | 14307 | -0.009526444042318906 | `multi_member_archive` |
| 9 | `experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/archive_nonidentity/archive.zip` | 202649 | 14262 | -0.009496480389428408 | `multi_member_archive` |
| 10 | `experiments/results/categorical_openpilot_payload_candidate_hardened_20260507_codex/archive.zip` | 179979 | 14183 | -0.009443877532131757 | `multi_member_archive` |

PR101 null-byte smoke archives are also in the top 25:

- `pr101_gold_master_gradient_null_byte_removal_smoke_20260521T010155Z/V_HALF/archive.zip`: 6,949 estimated recoverable bytes
- `pr101_gold_master_gradient_null_byte_removal_smoke_20260521T010155Z/V_ZERO/archive.zip`: 6,919 estimated recoverable bytes

Largest duplicate archive SHA clusters:

- 185 copies:
  `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`
- 120 copies:
  `e44937e29f4a937434a41e8aba4a28501494c6fb10bfac05f7fc2e0cdf10671b`
- 69 copies:
  `93e3ccad7b616898eb4db7a901d3f15adaff9e53371b08e61389c8313f8dc423`
- 60 copies:
  `dbd0d7fb31e6c270ccf09016bc8363d58e48e1593939531c32b34fce46531152`

## Interpretation

This does not reverse the archive-layer saturation finding for the PR101/FEC6
frontier. The full scan says a more precise thing:

1. Generic ZIP/member entropy headroom is not concentrated only in old public
   imports; there are recent LFV1/LAPose/HFV1 and PR101 null-byte candidate
   surfaces worth ranking.
2. The largest nominal headroom values are diagnostic, not score claims. They
   may be unreachable once runtime grammar, inflate fidelity, exact scorer
   behavior, and contest compliance are enforced.
3. The next frontier-moving artifact path should be archive-backed and
   candidate-specific: verify the top LFV1/HFV1/null-byte archives through
   inflate parity, scorer axis decomposition, and member-grammar recoding before
   writing another broad conceptual roadmap.

## Recommended next action

Build the next tool as a focused consumer of this inventory:

- input: `archive_surface_inventory.json`
- filter: recent LFV1/HFV1/PR101-null-byte rows with `score_claim=false`
- output: a ranked recode/eval queue with required checks:
  inflate parity, archive SHA, member SHA, runtime-tree custody, existing
  advisory/exact-eval artifacts, and whether the candidate is byte-closed

This keeps the next move on real bytes and avoids spending compute on ATW2 CDF
until full candidate archives exist.
