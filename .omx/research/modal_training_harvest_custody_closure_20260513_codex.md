# Modal training harvest custody closure (2026-05-13, codex)

## Summary

Bulk Modal training harvest was rerun with provider-custody hardening. The
initial audit found 60 `experiments/results/lane_*_modal` directories with
`modal_metadata.json` but without a complete harvested+terminal-marker closure.
After fixing the recovery helpers and rerunning `tools/harvest_modal_calls.py`,
the same audit reports:

```text
unclosed_or_unharvested 0
```

This is custody closure only. It creates no score claims, no rank/kill claims,
and no promotion authority.

## Fix

The bug class was marker-loss in already-harvested or legacy Modal training
recoveries:

- `append_modal_training_terminal_claim()` returned a non-appended manifest for
  missing legacy `lane_id` / job metadata but did not write the marker file.
- `append_platform_training_anchor()` returned non-appended cost manifests for
  missing or incomplete cost metadata but did not write the marker file.
- `tools/harvest_modal_calls.py` left provider lookup errors, expired caches,
  and legacy root-level `harvest_summary.json` files without a normalized
  `_harvest_summary.json` inside `harvested_artifacts/`.

The landed behavior writes explicit marker manifests for all of those paths.
Markers preserve `score_claim=false`, `promotion_eligible=false`, and, when
applicable, `rank_or_kill_eligible=false`.

## Evidence

Commands:

```bash
.venv/bin/python tools/harvest_modal_calls.py
.venv/bin/python - <<'PY'
from pathlib import Path
import json
root=Path('experiments/results')
rows=[]
for d in sorted(root.glob('lane_*_modal')):
    meta=d/'modal_metadata.json'
    if not meta.exists():
        continue
    harvested=(d/'harvested_artifacts').exists() and any((d/'harvested_artifacts').iterdir())
    term=(d/'modal_training_terminal_claim.json').exists()
    cost=(d/'cost_band_anchor_appended.json').exists()
    if not (harvested and term):
        m=json.loads(meta.read_text())
        rows.append((d.as_posix(), harvested, term, cost, m.get('label'), m.get('call_id')))
print('unclosed_or_unharvested', len(rows))
print(rows[:20])
PY
```

Result:

```text
Found 65 dispatched lanes with modal_metadata.json
unclosed_or_unharvested 0
CLAIM_SUMMARY active=0 stale_nonterminal=0 terminal_latest=688 unparsable_timestamp=0
```

Focused tests:

```bash
.venv/bin/pytest \
  src/tac/tests/test_modal_training_claims.py \
  src/tac/tests/test_modal_training_cost_anchor.py \
  src/tac/tests/test_modal_training_harvest_summary.py
```

Result:

```text
10 passed
```

## Score-lowering relevance

This closes stale provider ambiguity before the next exact-eval dispatch wave.
The actionable score-lowering queue remains unchanged: PR106/R2 PacketIR
identity parse/reemit and consumed-byte sidecar transforms first, then claimed
exact CUDA/CPU evaluations only when the archive/runtime packet is byte-closed.
