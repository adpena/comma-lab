# Codex Findings: OP-SYN-1 Extract-All Manifest Runner

timestamp_utc: 2026-05-18T20:29:47Z
agent: codex
source_design_memo: .omx/research/codex_routing_directive_op_syn_1_master_gradient_six_archive_extension_20260518.md
task_id: codex_routing_directive_op_syn_1_master_gradient_six_archive_extension_20260518::OP_SYN_1
score_claim: false
research_only: false

## Finding

OP-SYN-1 still cannot legally emit six master-gradient anchors. The parser
surface can detect PR106/DP1/PR107-family packets, but only
`fec6_fp11_selector`, `pr101_lc_v2`, and `a1_finetuned` have registered
gradient-projector authority. The correct next hardening slice was therefore
not "extract all anchors"; it was a batch xray runner that makes the remaining
projector blockers explicit and machine-readable.

## Change

`tools/extract_master_gradient.py` now supports:

```bash
.venv/bin/python tools/extract_master_gradient.py list-grammars
.venv/bin/python tools/extract_master_gradient.py extract-all \
  --manifest experiments/results/master_gradient_6_archive_batch_20260518/manifest.json \
  --output experiments/results/master_gradient_6_archive_batch_20260518/extract_all_layouts.json
```

The `extract-all` manifest runner:

- resolves relative archive paths relative to the manifest file;
- detects each archive grammar from bytes rather than trusting manifest claims;
- serializes layout custody and `projection_contract` per archive;
- marks supported grammars as `anchor_ready` but performs no anchor write;
- marks unsupported grammars as `detection_only_blocked` with the exact
  missing projector;
- optionally returns non-zero with `--strict` when any row is missing,
  mismatched, errored, or detection-only.

No call path writes `.omx/state/master_gradient_anchors.jsonl` from
`extract-all`.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_extract_master_gradient.py -q
.venv/bin/ruff check tools/extract_master_gradient.py src/tac/tests/test_extract_master_gradient.py
.venv/bin/ruff check --isolated --force-exclude --select F821 --ignore-noqa \
  --exclude experiments/archive --exclude experiments/results \
  src/ experiments/ submissions/robust_current/ scripts/ tools/
.venv/bin/python tools/extract_master_gradient.py list-grammars | .venv/bin/python -m json.tool >/dev/null
.venv/bin/python tools/extract_master_gradient.py extract-all --help
git diff --check -- tools/extract_master_gradient.py src/tac/tests/test_extract_master_gradient.py
```

All checks passed.

## Residual OP-SYN-1 Blockers

- `dp1_pretrained_driving_prior_schema_projector_missing`
- `pr106_format0d_primary_payload_projector_missing`
- `pr107_apogee_schema_projector_missing`

The prior `extract_all_batch_cli_missing` blocker is closed by this landing.
