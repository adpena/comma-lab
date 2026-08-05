# GB1 Receipt — EK1 #351 Guard Bundle Warn-Only Landing (2026-08-05)

## Answer First

GB1 landed the EK1 #351-adjacent producer-identity backfill guard as WARN-ONLY in `preflight_all()`.

Current-main census through the landed guard:

| guard | violations | denominator | strictness |
|---|---:|---:|---|
| `check_evidence_authority_claims_producer_identity_backfill_ready` | 188 | 381 canonical-equation `.py` files scanned | WARN-ONLY |

This exactly reproduces EK1's measured 188 live violations while correcting the denominator to the current scanner scope. No strict flip was performed.

STRICT-FLIP TRIGGER: `p0_332_provenance_bijection_backfill_20260717` / #670 owns the backfill. When the backfill drives this guard's live count to 0 on current main, flip the `preflight_all()` callsite from `strict=False` to `strict=True` in a separate reviewed landing.

No scorer job, no `evaluate.py`, no archive claim, no pointer movement.

## Landed Code

- `src/tac/preflight.py`
  - Added `_check_351_canonical_producer_identity_backfill_debt(...)`, a stricter #351 scanner that recursively scans `src/tac/canonical_equations/**/*.py`.
  - Added `check_evidence_authority_claims_producer_identity_backfill_ready(...)`, wired WARN-ONLY beside the existing #332/#351 provenance gates.
  - Kept the existing strict `check_evidence_authority_claims_are_custodied(...)` surface intact.
  - Allowed the existing strict #351 helper recognizer to accept the already-stronger GB1 exact-helper shape as a valid superset.
- `src/tac/tests/test_check_351_producer_identity_backfill_ready.py`
  - Added 25 tests covering positive, negative, waiver-respect, placeholder rejection, edge/candidate-scope, denominator, strict-mode future flip, and `preflight_all()` wiring behavior.

## Census Breakdown

Measured command:

```bash
.venv/bin/python - <<'PY'
from collections import Counter
from pathlib import Path
from tac import preflight
violations, denominator = preflight._check_351_canonical_producer_identity_backfill_debt(Path.cwd(), include_denominator=True)
print(len(violations), denominator)
PY
```

Per-defect breakdown from the 188 violations:

| defect | count |
|---|---:|
| missing fail-closed exact-path helper | 187 |
| provenance calls are not one exact canonical call per guarded producer | 185 |
| canonical_producers is not an exact tuple/list of guarded labels | 176 |
| verified producer inputs are not statically identifiable | 165 |
| source_receipt bypasses exact canonical-path binding | 19 |
| canonical_producers does not exactly match guarded labels in argument order | 11 |
| verified producer contains a non-SHA or unreachable explicit raise | 5 |
| provenance builder is not the canonical unshadowed import | 2 |
| source_manifest bypasses exact canonical-path binding | 1 |
| SOURCE_MEASUREMENT_SHA256 has a decoy or missing SHA use | 1 |
| SOURCE_FRONTIER_MAGNITUDE_SHA256 has a decoy or missing SHA use | 1 |
| receipt_path bypasses exact canonical-path binding | 1 |
| source_artifact bypasses exact canonical-path binding | 1 |

## RECALL EVIDENCE

Sources searched:

- Charter and common contract: `.omx/tmp/codex_runs/gb1_prompt.md`, `.omx/tmp/codex_runs/_common_contract.md`.
- Governing files: `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md` (byte-identical to `CLAUDE.md`), `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`.
- EK1 source receipt and manifest: `.omx/research/ddm_ek1_20260805/EK1_RECEIPT.md`, `.omx/research/ddm_ek1_20260805/ek1_residue_manifest.json`.
- Current code lineage: `src/tac/preflight.py`, `src/tac/confound_gates.py`, `src/tac/tests/test_check_351_canonical_producer_identity_scope_extension.py`, `src/tac/canonical_equations/einstein_kolmogorov_crux_20260719.py`.
- Residue source comparison: `.omx/tmp/codex_worktrees/einstein_kolmogorov_crux_20260719T212159Z/src/tac/preflight.py` and its #351 test file.
- Research/control-plane queries:
  - `rg -n "gb1|codex_runs|common_contract|gateway|lane|charter" /Users/adpena/.codex/memories/MEMORY.md`
  - `rg -n "GB1|EK1|Catalog #351|canonical producer|canonical_producers|p0_332|#670|producer identity|provenance bijection" ...`
  - `.venv/bin/python tools/list_canonical_equations.py --json`

Found beyond charter seeds:

- `.omx/state/operator_p0_ledger.jsonl` records `p0_332_provenance_bijection_backfill_20260717` as in-progress and owner-routed for the #332/#406 backfill. This changed the strict-flip trigger from a generic "future backfill" to the explicit #670 / `p0_332` owner/fire order.
- The EK1 worktree residue carried old unrelated preflight deltas, including top-level `torch` import and reversions of current-main hook work. This changed the plan from copy/merge to a scoped transplant of the #351 scanner only.
- The current Einstein-Kolmogorov runtime helper already has the stricter exact-path body; only the old strict static recognizer needed compatibility with that stronger shape.

## Verification

- PASS: `.venv/bin/python -m py_compile src/tac/preflight.py src/tac/tests/test_check_351_producer_identity_backfill_ready.py`
- PASS: `.venv/bin/python -m pytest src/tac/tests/test_check_351_producer_identity_backfill_ready.py` — 25 passed.
- PASS: existing strict static #351 scanner remains green: `_check_351_canonical_producer_identity(Path.cwd())` returned 0.
- PASS: `git diff --check -- src/tac/preflight.py src/tac/tests/test_check_351_producer_identity_backfill_ready.py`
- BOUNDARY: the broader legacy `test_check_351_canonical_producer_identity_scope_extension.py` file still has two runtime builder failures on this checkout because `.omx/research/einstein_kolmogorov_crux_measurement_20260719.json` has `st_nlink=3`; GB1 did not edit that canonical equation or evidence file.

Positive controls fired after the guard-order false positive was fixed: unrouted path arguments, missing exact helper, resolve-only alias laundering, guarded path not feeding provenance, guarded label not feeding `canonical_producers`, non-fail-closed SHA equality, wrong provenance/SHA pairing, renamed `_file` parameter, nested module scope, local provenance-builder shadowing, assignment shadowing, and placeholder waiver all produce violations.

## Catalog-Row Draft

Draft for MAIN to place; do not paste into `CLAUDE.md` until the operator chooses the catalog placement:

> **Catalog #351 scope extension — backfill-ready exact canonical-producer custody (GB1/EK1, 2026-08-05).** The existing evidence-authority umbrella now has a WARN-ONLY backfill scanner `check_evidence_authority_claims_producer_identity_backfill_ready` that recursively audits canonical-equation producer builders for exact helper shape, canonical unshadowed provenance-builder import, guarded path consumption by provenance, guarded label emission through `canonical_producers`, fail-closed SHA/provenance pairing, and same-line `CANONICAL_PRODUCER_IDENTITY_OK:<rationale>` waivers with placeholder rejection. Live count at landing: 188 violations / 381 canonical-equation `.py` files scanned. Strict-flip trigger: #670 / `p0_332_provenance_bijection_backfill_20260717` drives live count to 0, then flips the callsite in a separate reviewed landing. This is a Catalog #351 scope extension, not a new catalog number, per Catalog #299 consolidation discipline.

## Follow-Ons

- QUEUED-WITH-FIRE-ORDER: #670 / `p0_332_provenance_bijection_backfill_20260717` burns down or substantively waives the 188 rows, reruns the denominatored census, and strict-flips only at live count 0.
- FOLDED: no CLAUDE.md edit in this landing; catalog row remains drafted here for MAIN.
- FOLDED: no scorer/eval launch; charter was scorer-free and $0.

## NEXT_IF_RESUMED

1. Do not strict-flip this guard until the census returns 0 violations on current main.
2. Backfill rows by making each verified producer use the exact helper, consume guarded paths in provenance, emit guarded labels in `canonical_producers`, and pair every source SHA to the correct provenance object.
3. For any intentional exception, add same-line `# CANONICAL_PRODUCER_IDENTITY_OK:<substantive rationale>` on the producer definition line; placeholders must not waive.
4. Rerun the GB1 test file and the live census with denominator before attempting the strict flip.

```json
{
  "schema": "pact.gb1_receipt.v1",
  "arm": "gb1",
  "date": "2026-08-05",
  "guard": "check_evidence_authority_claims_producer_identity_backfill_ready",
  "strictness": "warn_only",
  "violations": 188,
  "denominator_files_scanned": 381,
  "strict_flip_trigger": "p0_332_provenance_bijection_backfill_20260717 / #670 live count reaches 0",
  "scorer_job": false,
  "evaluate_py_run": false,
  "pointer_moved": false,
  "tests": {
    "py_compile": "pass",
    "gb1_guard_pytest": "25 passed",
    "existing_strict_static_351": "0 violations"
  }
}
```

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
