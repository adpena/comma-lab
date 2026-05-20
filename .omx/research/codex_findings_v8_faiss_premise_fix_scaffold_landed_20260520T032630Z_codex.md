# Codex Findings - V8 Faiss Premise Fix Scaffold Landed

**UTC:** 2026-05-20T03:26:30Z  
**Owner:** codex  
**Lane:** `lane_v8_learned_compression_faiss_scaffold_codex_20260520`  
**Source design:** `.omx/research/v1_faiss_v8_learned_compression_faiss_design_20260519.md`  
**Intake:** `.omx/research/codex_findings_latest_design_memo_implementation_intake_20260520T031038Z_codex.md`  
**Score claim:** none  
**Promotion eligible:** false  
**Ready for exact eval dispatch:** false  
**Provider spend:** none

## Verdict

`PROCEED_TO_NEXT_LOCAL_IMPLEMENTATION_GATE`, not paid dispatch.

The V8 learned-compression Faiss design is now represented by importable local
scaffold code, a fail-closed submission runtime shell, a disabled operator
recipe, and focused tests. The design's stale helper premises are corrected in
code without mutating the historical design memo.

## Implementation Landed

1. Rehomed the PQ mutual-information verdict into reusable `tac` code:
   - `src/tac/optimization/faiss_ivf_pq_atw_channel.py`
   - new public symbols: `PqMiVerdict`, `compute_pq_mi_verdict`,
     `DEFAULT_MEANINGFUL_MI_THRESHOLD_BITS`, `INDEPENDENCE_TOLERANCE_BITS`
   - `tools/probe_atw_v2_1_faiss_pq_disambiguator.py` now consumes the canonical helper instead of carrying a tool-local implementation.

2. Added the V8 learned-compression Faiss scaffold:
   - `experiments/train_substrate_v8_learned_compression_faiss.py`
   - smoke mode writes `v8_smoke_results.json`
   - full mode raises `NotImplementedError` until categorical posterior, scale hyperprior, byte-closed export, and scorer-roundtrip training land together.

3. Added fail-closed submission/runtime surface:
   - `submissions/v8_learned_compression_faiss/inflate.py`
   - `submissions/v8_learned_compression_faiss/inflate.sh`
   - runtime validates `V8FAISS1` header and then refuses inflate because the decoder is not byte-closed yet.

4. Added disabled operator recipe:
   - `.omx/operator_authorize_recipes/substrate_v8_learned_compression_faiss_modal_a100_smoke.yaml`
   - `research_only: true`
   - `dispatch_enabled: false`
   - explicit dispatch blockers for all missing full-stack pieces.

5. Added tests:
   - `src/tac/tests/test_v8_learned_compression_faiss_scaffold.py`
   - extended `src/tac/tests/test_atw_v2_1_faiss_ivf_pq.py`

6. Hardened Catalog #287 phantom-helper scan:
   - appended exact waiver-manifest rows to `.omx/state/catalog_287_phantom_api_waivers.jsonl`
   - preserved append-only historical memos instead of rewriting them
   - re-ran strict scan to `catalog-287-strict-ok`

## Premise Corrections

| Memo premise | Corrected implementation authority |
|---|---|
| `tac.substrates._shared.score_aware_common.score_pair_components` | `tac.substrates.score_aware_common.score_pair_components_dispatch` |
| `compute_pq_mi_verdict` as if it were a reusable helper | now importable from `tac.optimization.faiss_ivf_pq_atw_channel.compute_pq_mi_verdict` |
| `tac.provenance.build_provenance_for_contest_archive_byte_member` | `tac.provenance.build_provenance_for_archive_member` |

These are premise fixes and scaffold integration, not changes to Claude's
historical design memo. The original memo remains append-only provenance.

## Smoke Artifact

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/train_substrate_v8_learned_compression_faiss.py \
  --output-dir experiments/results/lane_v8_learned_compression_faiss_smoke_codex_20260520T032000Z \
  --smoke \
  --num-pairs 4 \
  --categorical-groups 8 \
  --codebook-size 16
```

Artifact:

- `experiments/results/lane_v8_learned_compression_faiss_smoke_codex_20260520T032000Z/v8_smoke_results.json`
- SHA-256: `ce5cb47aa95d5d30d649f02fca869d221e2235b307924e371c24c5e1c544c82e`

Manifest authority:

- `research_only=true`
- `dispatch_enabled=false`
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- axis: `[diagnostic-CPU scaffold smoke; no scorer load]`

## Recursive Senior Review

Contest compliance: PASS for current scope. No score was claimed, no archive was promoted, no exact eval was invoked, no provider was dispatched, and the operator recipe refuses dispatch.

1:1 conformance: PASS for current scope. The runtime surface preserves the `inflate.sh archive_dir output_dir file_list` shape, but fails closed because the V8 decoder is not implemented. This avoids false 1:1 contest authority.

Helper authority: PASS. The new smoke manifest cites importable helpers only for corrected surfaces. The stale design references are recorded as corrected premises, not live helper authority.

Sister-WIP isolation: PASS. This landing avoids active PR101/FEC6 submission-runtime and compliance files owned by `claude_slot_rr_d3_compliance_gate_clearance_20260520`.

RACE_MODE handling: PARTIAL-BY-DESIGN. `RACE_MODE_ACTIVE.flag` exists, so the scaffold names the actuator recipe and smoke/full handoff surfaces. The recipe remains disabled because the local full-stack archive/export/training path is not implemented.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_atw_v2_1_faiss_ivf_pq.py \
  src/tac/tests/test_probe_atw_v2_1_faiss_pq_disambiguator.py \
  src/tac/tests/test_v8_learned_compression_faiss_scaffold.py \
  -p no:cacheprovider
```

Result: `43 passed in 0.50s`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/operator_authorize.py \
  --recipe substrate_v8_learned_compression_faiss_modal_a100_smoke \
  --dry-run \
  --target none
```

Result: dry-run refusal on `dispatch_enabled=false` plus the four V8 blockers.

```bash
.venv/bin/python tools/lane_maturity.py validate
```

Result: `OK -- 1037 lane(s) validated cleanly.`

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/canonical_task_status.py --validate
```

Result: `{"rows": 242, "status": "valid"}` before this memo's terminal task row.

```bash
git diff --check -- <V8 touched files>
```

Result: no whitespace errors.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
from pathlib import Path
import importlib.util
spec = importlib.util.spec_from_file_location('preflight', 'src/tac/preflight.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.check_no_docstring_overstatement_without_evidence_tag(
    repo_root=Path('.'),
    strict=True,
    scan_research_memos=True,
    scan_memory_files=False,
)
print('catalog-287-strict-ok')
PY
```

Result: `catalog-287-strict-ok`.

## Remaining Blockers

- `gumbel_softmax_categorical_posterior_not_implemented`
- `scale_hyperprior_decoder_not_implemented`
- `byte_closed_export_not_implemented`
- `score_aware_eval_roundtrip_training_not_implemented`
- `modal_dispatch_recipe_is_research_only_and_disabled`

## Next Codex Gate

Implement the smallest byte-closed V8 archive grammar and synthetic decoder
roundtrip next. Do not enable paid dispatch until:

1. local V8 archive build emits a single contest member;
2. `inflate.sh` consumes that member into deterministic raw frames or a
   documented fail-closed blocker;
3. score-aware eval-roundtrip training path exists behind explicit mode flags;
4. `operator_authorize.py --dry-run` shows dispatch blockers cleared;
5. lane dispatch is claimed before any provider call.

## 6-Hook Wire-In

1. Sensitivity map: deferred until real V8 codeword stream exists.
2. Pareto constraint: scaffold manifest records byte estimate and non-promotable status.
3. Bit allocator: deferred until categorical stream grammar lands.
4. Cathedral autopilot dispatch: disabled recipe exists and refuses dispatch.
5. Continual-learning posterior: no empirical score anchor; no posterior score write.
6. Probe disambiguator: `compute_pq_mi_verdict` is canonicalized for V8/V4/V1-V3 reuse.
