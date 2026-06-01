# Tensor Payload Grammar Operator-Briefing Wire-In

- **UTC:** 2026-06-01T19:44:50Z
- **Author:** Codex
- **Axis:** `[planning-only byte-profile]`
- **Score claim:** `false`
- **Promotion eligible:** `false`
- **Ready for exact eval dispatch:** `false`

## Finding

The generic tensor payload grammar optimizer and cathedral consumer are now
discoverable from the normal operator briefing surface. This closes the
artifact-only gap for the optimal-grammar lane: per-tensor grammar reports now
show up as a planning-only phase that tells the campaign loop whether a tensor
family is saturated, weak-gap, or worth receiver/archive binding.

Current real-artifact summary from `/Volumes/VertigoDataTier/pact`:

- `artifact_count`: `6`
- `consumer_result_count`: `3`
- `status`: `SATURATED`
- `receiver_work_justified_count`: `0`
- `demotion_recommended_count`: `3`
- `score_claim`: `false`
- `ready_for_exact_eval_dispatch`: `false`

That matches the PR101/PACT-NeRV evidence: current competitive tensor grammar
families are near entropy floor, so repeated same-substrate format churn should
be demoted unless a new substrate/export produces an unsaturated entropy gap.

## Landing

Code:

- `tools/operator_briefing.py`
- `src/tac/tests/test_operator_briefing.py`

New operator surface:

- JSON key: `tensor_payload_grammar`
- readiness key: `phase_6c_tensor_payload_grammar`
- text section: `Phase 6c.1 — Generic tensor payload grammar`

The scanner is schema-gated and bounded. It scans repo-local artifacts and the
SSD artifact roots:

- `/Volumes/VertigoDataTier/pact`
- `/Volumes/APDataStore/pact`

It consumes:

- `tensor_payload_grammar_optimizer.v1`
- `tensor_payload_grammar_consumer_result.v1`

## Verification

```bash
uv run ruff check tools/operator_briefing.py src/tac/tests/test_operator_briefing.py src/tac/tests/test_tensor_payload_grammar_consumer.py src/tac/tests/test_tensor_payload_grammar_optimizer.py
uv run pytest -q src/tac/tests/test_operator_briefing.py -k 'tensor_payload' src/tac/tests/test_tensor_payload_grammar_consumer.py src/tac/tests/test_tensor_payload_grammar_optimizer.py src/tac/tests/test_section_payload_grammar_optimizer.py
uv run python - <<'PY'
import importlib.util, json, sys
from pathlib import Path
path = Path("tools/operator_briefing.py")
spec = importlib.util.spec_from_file_location("operator_briefing_live_check", path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)
summary = mod._tensor_payload_grammar_summary()
print(json.dumps({
    "status": summary.get("status"),
    "artifact_count": summary.get("artifact_count"),
    "consumer_result_count": summary.get("consumer_result_count"),
    "receiver_work_justified_count": summary.get("receiver_work_justified_count"),
    "demotion_recommended_count": summary.get("demotion_recommended_count"),
    "ready_for_exact_eval_dispatch": summary.get("ready_for_exact_eval_dispatch"),
    "score_claim": summary.get("score_claim"),
}, sort_keys=True))
PY
```

Results:

- `ruff`: pass
- focused tests: `10 passed, 59 deselected`
- live summary: `{"artifact_count": 6, "consumer_result_count": 3, "demotion_recommended_count": 3, "ready_for_exact_eval_dispatch": false, "receiver_work_justified_count": 0, "score_claim": false, "status": "SATURATED"}`

## Next

The optimal-grammar lane should now use the briefing result as a default
campaign diagnostic:

- `SATURATED`: demote same-substrate format churn and route effort to a new
  score-native carrier or train/export lane.
- `NEEDS_RECEIVER_BINDING`: bind the selected tensor grammar map into the
  substrate receiver/archive, then require byte-closed replay before any exact
  auth dispatch.

