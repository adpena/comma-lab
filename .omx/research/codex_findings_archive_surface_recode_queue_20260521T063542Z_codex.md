# Codex Findings: Archive Surface Recode Queue

- timestamp_utc: 2026-05-21T06:35:42Z
- lane: archive_surface_inventory_consumer
- status: LANDED_STRICT_QUEUE_PLANNER_PLUS_FULL_QUEUE
- score_claim: false
- promotion_eligible: false
- ready_for_exact_eval_dispatch: false

## What landed

New tool:

- `tools/plan_archive_surface_recode_queue.py`

The tool consumes `archive_surface_inventory.json` and emits a ranked, scoped
queue for candidate-specific recode/eval follow-up. The queue keeps ordering
explicit:

1. `pr101_null_byte_smoke`
2. `hfv1_pr101_adapter`
3. `lfv1_lapose_foveation`
4. `openpilot_prior_candidate`
5. `z7_world_model_candidate`

Within each class, rows are ordered by estimated recoverable ZIP bytes, archive
bytes, and path. The queue is diagnostic only: every row remains
`score_claim=false`, `promotion_eligible=false`, and
`ready_for_exact_eval_dispatch=false`.

## Guardrail fixed during implementation

The first planner draft over-scoped evidence discovery and could borrow
`contest_auth_eval.json`, `report.txt`, or parity artifacts from sibling
variants. That would have made blocked candidates appear promotion-ready.

Fix landed in the tool:

- evidence lookup is direct-only;
- ancestor search is capped at the candidate/experiment-root depth;
- no repo-root `report.txt` or sibling-variant evidence can satisfy a row;
- candidate manifests such as `archive_member_manifest.json`, `manifest.json`,
  and `readiness.json` are tracked separately from exact eval artifacts.

This matters because the honest queue result is not "ready"; it is "which
single blocker should be removed first."

## Verification commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/plan_archive_surface_recode_queue.py \
  --inventory-json experiments/results/archive_surface_inventory_20260521T062618Z/archive_surface_inventory.json \
  --output-dir experiments/results/archive_surface_recode_queue_smoke3_20260521T0650Z \
  --limit 15 >/tmp/archive_surface_recode_queue_smoke3.json

git diff --check -- tools/plan_archive_surface_recode_queue.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py scan

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file \
  tools/plan_archive_surface_recode_queue.py --status reviewed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check \
  tools/plan_archive_surface_recode_queue.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/plan_archive_surface_recode_queue.py \
  --inventory-json experiments/results/archive_surface_inventory_20260521T062618Z/archive_surface_inventory.json \
  --output-dir experiments/results/archive_surface_recode_queue_20260521T063508Z \
  --limit 25 > experiments/results/archive_surface_recode_queue_20260521T063508Z/stdout.json
```

Review tracker result:

- `tools/plan_archive_surface_recode_queue.py`: NORMAL
- 18 entities reviewed
- 0 policy violations

## Queue artifacts

- JSON: `experiments/results/archive_surface_recode_queue_20260521T063508Z/archive_surface_recode_queue.json`
- Markdown: `experiments/results/archive_surface_recode_queue_20260521T063508Z/archive_surface_recode_queue.md`
- stdout mirror: `experiments/results/archive_surface_recode_queue_20260521T063508Z/stdout.json`

SHA-256:

```text
1125a3bc31279aa5d859ecfcdd8ec0a73f43084603a6a0982dd2a3c92afe971b  archive_surface_recode_queue.json
ed94af524390a5ed0be2d643f408e8ac2bfc52249cbfa50f7c4a1e7b2c5873dd  archive_surface_recode_queue.md
1125a3bc31279aa5d859ecfcdd8ec0a73f43084603a6a0982dd2a3c92afe971b  stdout.json
```

Byte counts:

```text
 69876  archive_surface_recode_queue.json
  9782  archive_surface_recode_queue.md
 69876  stdout.json
```

## Full-queue result

- Inventory rows consumed: 3,785
- Rows after current frontier-relevant class filter: 78
- Rows after duplicate archive SHA removal: 49
- Dropped duplicate archive rows: 29
- Emitted queue rows: 25

Class distribution in emitted queue:

- `z7_world_model_candidate`: 10
- `hfv1_pr101_adapter`: 5
- `pr101_null_byte_smoke`: 4
- `lfv1_lapose_foveation`: 4
- `openpilot_prior_candidate`: 2

Promotion-blocker distribution:

- 24 rows: `missing_contest_auth_eval_json`
- 20 rows: `missing_inflate_runtime`
- 20 rows: `missing_report_txt`
- 20 rows: `missing_inflate_parity_or_output_manifest`
- 20 rows: `not_byte_closed_submission_candidate`
- 14 rows: `missing_archive_or_candidate_manifest`

Rows with exactly one blocker:

| rank | class | recoverable bytes | only blocker | archive |
|---:|---|---:|---|---|
| 6 | `hfv1_pr101_adapter` | 14307 | `missing_contest_auth_eval_json` | `experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/archive_identity/archive.zip` |
| 7 | `hfv1_pr101_adapter` | 14262 | `missing_contest_auth_eval_json` | `experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/archive_nonidentity/archive.zip` |
| 8 | `hfv1_pr101_adapter` | 13981 | `missing_contest_auth_eval_json` | `experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/advisory_raw_eval/seed_top16_component_hardpairs/archive.zip` |
| 9 | `hfv1_pr101_adapter` | 30 | `missing_contest_auth_eval_json` | `experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/advisory_raw_eval/no_sidecar/archive.zip` |

## Interpretation

The queue changes the next move. The largest nominal byte headroom is LFV1, but
the most mature next frontier-moving surface is the HFV1 PR101 adapter family:
it already has archive SHA verification, runtime/report/manifest evidence, and
local inflate/parity evidence. Its remaining blocker in this queue is exact
auth-eval harvest/adjudication (`contest_auth_eval.json`).

The PR101 null-byte smoke rows remain important but are not byte-closed
candidate packets in this local evidence view. They need runtime/manifest/parity
closure before another score-facing step.

## Recommended next action

Do not launch a broad new substrate sweep from this evidence. The next concrete
step is to build or invoke a focused HFV1 PR101 adapter exact-eval readiness
actuator for the three nontrivial one-blocker rows:

- `archive_identity/archive.zip`
- `archive_nonidentity/archive.zip`
- `advisory_raw_eval/seed_top16_component_hardpairs/archive.zip`

That actuator should perform the dispatch-claim lifecycle, recompute
archive/member hashes, bind runtime-tree custody, and either produce
`contest_auth_eval.json` or record the exact missing dispatch precondition.
