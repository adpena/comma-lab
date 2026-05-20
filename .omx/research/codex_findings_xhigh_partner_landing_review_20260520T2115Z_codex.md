# Codex findings: xhigh partner landing review

Date: 2026-05-20T21:15Z
Reviewer: Codex xhigh adversarial review worker
Scope: read-only review of fast-moving partner landings in `/Users/adpena/Projects/pact`.
Write scope honored: this memo only. No code, PR body, README, archive, manifest, or partner file was edited.

## Findings by severity

### HIGH - tracked modules now depend on untracked modules; partial commit will break clean checkouts

Evidence:
- `src/tac/atom/__init__.py` now imports `.contest_granularity` at lines 99+ in the dirty diff, while `src/tac/atom/contest_granularity.py` and `src/tac/atom/tests/test_contest_granularity.py` are untracked.
- `src/tac/cathedral_consumers/top_k_byte_sensitivity_consumer/__init__.py` now imports `tac.optimization.byte_score_impact` at line 80 in the dirty diff, while `src/tac/optimization/byte_score_impact.py` is untracked.
- Additional untracked consumers depend on these files: `tools/build_contest_atom_lattice.py`, `tools/plan_topk_byte_score_impact_targets.py`, and `src/tac/tests/test_byte_score_impact.py`.

Current dirty-tree imports pass, so this is not a current local import failure. The risk is transactional: committing only the modified tracked files without the new untracked dependencies would make `import tac.atom` and `import tac.cathedral_consumers.top_k_byte_sensitivity_consumer` fail in a clean checkout.

Recommended fix:
- Land the tracked import edits and their untracked dependency modules/tests/tools in the same canonical serializer batch, or defer the import re-exports until the new modules are committed.
- Before committing, run `git status --short -- src/tac/atom src/tac/optimization src/tac/cathedral_consumers/top_k_byte_sensitivity_consumer tools/build_contest_atom_lattice.py tools/plan_topk_byte_score_impact_targets.py src/tac/tests/test_byte_score_impact.py` and verify no dependency is left untracked.

### HIGH - PR110-linked public documentation still has AI-attribution/transitive-surface risk

Evidence:
- `docs/meta_engineering_vision.md:79` links `docs/ai_assisted_inverse_steganalysis_persona_council.md` and explicitly describes the named-persona council methodology as grounded in Anthropic persona-vectors research.
- The existing PR110 audit memos already flag this exact transitive-doc class as risky for public PR surfaces.
- Direct grep of the dirty submission README/manifest did not find `Claude`, `Anthropic`, `Co-Authored`, `claude.com`, or `anthropic.com`; the risk is the linked docs surface, not the local PR README/manifest diff itself.

Recommended fix:
- Do not point live PR110 or public submission surfaces at docs containing Claude/Codex/Anthropic/persona-process material.
- Keep that methodology material in internal `.omx/research` or a private appendix unless the operator explicitly chooses to publish it.

### MEDIUM - selector V2 docstring labels a pre-training prediction as `[contest-CPU]`

Evidence:
- `src/tac/substrates/pact_nerv_selector_v2/__init__.py:18-23` says the rate-axis prediction is ``-30..-100 bytes / +0.000..-0.003 [contest-CPU]``.
- The same file says the lane is `L0 SKETCH` and `research_only=true`, and the recipe is correctly `dispatch_enabled: false` / `research_only: true`.

This is axis-label drift: a pre-training byte/rate prediction should not carry `[contest-CPU]` until exact Linux x86_64 archive/runtime custody exists for that candidate. The surrounding recipe is safe, but the docstring can seed false authority in later summaries.

Recommended fix:
- Relabel that phrase to `[predicted]` or `[rate-model prediction; no contest score]`.
- Keep `[contest-CPU]` reserved for exact upstream evaluator runs on the exact archive/runtime packet.

### MEDIUM - `src/tac/bit_allocator.py` deletion leaves stale path references

Evidence:
- The index stages deletion of `src/tac/bit_allocator.py` and `src/tac/tests/test_bit_allocator.py`.
- The package replacement exports the legacy API successfully from `src/tac/bit_allocator/__init__.py`, and focused tests pass.
- Stale metadata still names the deleted path: `src/tac/solvers/more_optimal_algorithms.py:210`, `:317`, and `src/tac/tests/test_more_optimal_algorithms_shim.py:161-174` use `src/tac/bit_allocator.py::allocate_bits`.

This is not a runtime import break in the current tree, but it creates dead callsite links in atoms/metadata after the file deletion lands.

Recommended fix:
- Update metadata/test callsite strings to `src/tac/bit_allocator/lane_omega.py::allocate_bits` or the import-level API `tac.bit_allocator.allocate_bits`, depending on whether the field is meant to be source-file or Python-symbol addressable.

### LOW - generated artifacts are present; keep them out of commits unless explicitly curated

Evidence:
- `__pycache__` / `.pyc` files exist under the new selector and PACT-NERV variant trees, but `git check-ignore` confirms they are ignored by `.gitignore`.
- `experiments/results/_modal_harvest_summary.json` is tracked and dirty with a large generated summary diff.

Recommended fix:
- Do not add any `__pycache__` files.
- Confirm `_modal_harvest_summary.json` is intended durable state before including it in a landing. If it is provider/result churn, distill the useful signal into a dated `.omx/research` memo or a small committed manifest and leave the rebuildable/generated summary out.

## Evidence

- Branch/source of truth: `git branch --show-current` returned `main`.
- Worktree state: `git status --short --branch` shows `main...origin/main [ahead 48]` with a large dirty set and staged deletion of `src/tac/bit_allocator.py` / `src/tac/tests/test_bit_allocator.py`.
- Active partner-owned surfaces from `.omx/state/subagent_progress.jsonl` tail:
  - bit allocator package move / OP3 downstream wire-in.
  - per-byte methodology docs + auto-trigger cathedral consumer + `reports/latest.md`.
  - PACT-NERV selector V2/V3/V4 packages and train/remote/recipe files.
  - PACT-NERV G2 mid-LOC variants: distilled_scorer, VQ, Bayesian, multi_modal, diffusion_trajectory.
  - PR110 writeup decision symposium writing only `.omx/research/...` and `/tmp/...`.
- Direct import smoke passed for `tac.bit_allocator`, `tac.atom`, and `top_k_byte_sensitivity_consumer.rank_archive_bytes_by_score_impact`.
- Focused tests passed:
  - bit allocator / atom / OP3 export slice: `91 passed`.
  - selector V2/V3/V4 tests: `44 passed`.
  - PACT-NERV G2 variant package tests: `61 passed`.
- `git diff --check` and `git diff --cached --check` returned clean.
- Dirty submission README/manifest grep found no direct `Claude`, `Anthropic`, `Co-Authored`, `claude.com`, or `anthropic.com` tokens.

## Immediate commands main thread should run before landing more work

```bash
git status --short --branch
git diff --check
git diff --cached --check

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q \
  src/tac/bit_allocator/tests \
  src/tac/atom/tests/test_contest_granularity.py \
  src/tac/tests/test_byte_score_impact.py \
  src/tac/tests/test_op3_downstream_wire_in_t4_anchor.py::test_bit_allocator_exports_from_master_gradient_anchor_helper \
  src/tac/tests/test_op3_downstream_wire_in_t4_anchor.py::test_bit_allocator_module_all_includes_new_helper

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q \
  src/tac/substrates/pact_nerv_selector_v2/tests \
  src/tac/substrates/pact_nerv_selector_v3/tests \
  src/tac/substrates/pact_nerv_selector_v4/tests \
  src/tac/substrates/pact_nerv_distilled_scorer/tests \
  src/tac/substrates/pact_nerv_vq/tests \
  src/tac/substrates/pact_nerv_bayesian/tests \
  src/tac/substrates/pact_nerv_multi_modal/tests \
  src/tac/substrates/pact_nerv_diffusion_trajectory/tests

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
from tac.bit_allocator import allocate_bits, allocate_per_byte_from_master_gradient_anchor
from tac.atom import ContestAtom, ByteScope, ScoreVector, BudgetVector, ContestScopeKind
from tac.cathedral_consumers.top_k_byte_sensitivity_consumer import rank_archive_bytes_by_score_impact
print("imports ok")
PY

rg -n "Claude|Anthropic|Co-Authored|claude\\.com|anthropic\\.com" \
  experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/README.md \
  experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive_manifest.json \
  docs/meta_engineering_vision.md docs/ai_assisted_inverse_steganalysis_persona_council.md

git status --short -- \
  src/tac/atom src/tac/optimization src/tac/cathedral_consumers/top_k_byte_sensitivity_consumer \
  tools/build_contest_atom_lattice.py tools/plan_topk_byte_score_impact_targets.py src/tac/tests/test_byte_score_impact.py
```

For a full landing gate after the active partner batch stabilizes, run the repo's strict preflight entrypoint rather than inventing a new flag surface:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from tac.preflight import preflight_all
preflight_all()
PY
```

## Commands run

```bash
pwd
git status --short --branch
git branch --show-current
git diff --stat
git diff --cached --stat
git diff --cached --name-status
git ls-files src/tac/bit_allocator.py src/tac/bit_allocator src/tac/tests/test_bit_allocator.py src/tac/bit_allocator/tests
tail -n 120 .omx/state/subagent_progress.jsonl
tail -n 45 .omx/state/subagent_progress.jsonl
sed -n '1,260p' AGENTS.md
sed -n '1,260p' CLAUDE.md
sed -n '1,220p' PROGRAM.md
sed -n '1,80p' ~/.claude/projects/-Users-adpena-Projects-pact/memory/MEMORY.md
find .omx/research -maxdepth 1 -type f -name '*_directive_*' -mtime -1 -print | sort
sed -n '1,120p' .omx/research/codex_findings_sparse_residual_oracle_charged_smoke_20260520T204441Z_codex.md
sed -n '1,120p' .omx/research/council_t3_grand_strategy_review_20260520T120000Z.md
sed -n '1,120p' .omx/research/codex_routing_directive_task_triage_batch_20260520T120607Z.md
rg -n "Pact|bit_allocator|PR110|partner|frontier|public" /Users/adpena/.codex/memories/MEMORY.md
rg -n "NON-NEGOTIABLE|PR110|PR 110|live PR|contest-CUDA|contest-CPU|Main-Branch|main branch|bit_allocator|Catalog #220|Catalog #325|Catalog #335|Catalog #341" CLAUDE.md AGENTS.md PROGRAM.md
ls -la src/tac/bit_allocator src/tac/atom src/tac/substrates | sed -n '1,180p'
sed -n '1,240p' src/tac/bit_allocator/__init__.py
git show HEAD:src/tac/bit_allocator.py | sed -n '1,240p'
rg -n "tac\\.bit_allocator|from tac\\.bit_allocator|import tac\\.bit_allocator|allocate_bits\\(|BitAllocation|AllocationResult|round10|bit_allocator\\.py|test_bit_allocator" -S . --glob '!*.pyc' --glob '!*.jsonl' --glob '!*.log'
find src/tac/substrates/pact_nerv_selector_v2 src/tac/substrates/pact_nerv_selector_v3 src/tac/substrates/pact_nerv_selector_v4 -maxdepth 2 -type f | sort
git diff -- src/tac/atom/__init__.py
git diff -- src/tac/cathedral_consumers/top_k_byte_sensitivity_consumer/__init__.py
sed -n '1,260p' src/tac/atom/contest_granularity.py
rg -n "contest-CUDA|contest-CPU|macOS-CPU|advisory|promotable|score_claim|promotion_eligible|axis_tag|predicted|dispatch_enabled|research_only|archive_bytes|score" src/tac/substrates/pact_nerv_selector_v2 src/tac/substrates/pact_nerv_selector_v3 src/tac/substrates/pact_nerv_selector_v4 experiments/train_substrate_pact_nerv_selector_v2.py experiments/train_substrate_pact_nerv_selector_v3.py experiments/train_substrate_pact_nerv_selector_v4.py scripts/remote_lane_substrate_pact_nerv_selector_v2.sh scripts/remote_lane_substrate_pact_nerv_selector_v3.sh scripts/remote_lane_substrate_pact_nerv_selector_v4.sh .omx/operator_authorize_recipes/substrate_pact_nerv_selector_v2_modal_t4_dispatch.yaml .omx/operator_authorize_recipes/substrate_pact_nerv_selector_v3_modal_t4_dispatch.yaml .omx/operator_authorize_recipes/substrate_pact_nerv_selector_v4_modal_t4_dispatch.yaml
rg -n "Claude|Anthropic|Co-Authored|claude\\.com|anthropic\\.com" experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/README.md experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive_manifest.json .omx/research/pr110* reports/latest.md docs/per_byte_sensitivity_comparative_analysis_methodology.md docs/meta_engineering_vision.md
find src/tac/substrates/pact_nerv_selector_v2 src/tac/substrates/pact_nerv_selector_v3 src/tac/substrates/pact_nerv_selector_v4 -type f \( -name '*.pyc' -o -name '.DS_Store' \) -print
nl -ba src/tac/substrates/pact_nerv_selector_v2/__init__.py | sed -n '1,140p'
nl -ba src/tac/substrates/pact_nerv_selector_v3/__init__.py | sed -n '1,120p'
nl -ba src/tac/substrates/pact_nerv_selector_v4/__init__.py | sed -n '1,120p'
nl -ba .omx/operator_authorize_recipes/substrate_pact_nerv_selector_v2_modal_t4_dispatch.yaml | sed -n '1,170p'
nl -ba src/tac/solvers/more_optimal_algorithms.py | sed -n '190,330p'
nl -ba src/tac/tests/test_more_optimal_algorithms_shim.py | sed -n '150,210p'
nl -ba docs/meta_engineering_vision.md | sed -n '60,95p'
git status --short -- src/tac/optimization/byte_score_impact.py src/tac/atom/contest_granularity.py src/tac/atom/tests/test_contest_granularity.py src/tac/cathedral_consumers/top_k_byte_sensitivity_consumer/__init__.py src/tac/atom/__init__.py experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/README.md experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive_manifest.json experiments/results/_modal_harvest_summary.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
from tac.bit_allocator import allocate_bits, allocation_report, allocate_per_byte_from_master_gradient_anchor
from tac.atom import ContestAtom, ByteScope, ScoreVector, BudgetVector, ContestScopeKind
from tac.cathedral_consumers.top_k_byte_sensitivity_consumer import rank_archive_bytes_by_score_impact
print("imports ok")
PY
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q src/tac/bit_allocator/tests/test_lane_omega.py src/tac/bit_allocator/tests/test_per_byte_per_class_per_axis_pareto_dual.py src/tac/atom/tests/test_contest_granularity.py src/tac/tests/test_op3_downstream_wire_in_t4_anchor.py::test_bit_allocator_exports_from_master_gradient_anchor_helper src/tac/tests/test_op3_downstream_wire_in_t4_anchor.py::test_bit_allocator_module_all_includes_new_helper
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q src/tac/substrates/pact_nerv_selector_v2/tests/test_pact_nerv_selector_v2.py src/tac/substrates/pact_nerv_selector_v3/tests/test_pact_nerv_selector_v3.py src/tac/substrates/pact_nerv_selector_v4/tests/test_pact_nerv_selector_v4.py
git ls-files src/tac/substrates/pact_nerv_selector_v2 src/tac/substrates/pact_nerv_selector_v3 src/tac/substrates/pact_nerv_selector_v4 | rg "__pycache__|\\.pyc$" || true
git diff -- experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/README.md experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive_manifest.json | sed -n '1,240p'
nl -ba src/tac/optimization/byte_score_impact.py | sed -n '1,260p'
nl -ba src/tac/atom/contest_granularity.py | sed -n '1,260p'
git diff --check
git diff --cached --check
git ls-files experiments/results/_modal_harvest_summary.json .omx/state/canonical_task_status.jsonl .omx/state/modal_call_id_ledger.jsonl .omx/state/lane_registry.json .omx/state/lane_maturity_audit.log reports/cathedral_autopilot_evidence.jsonl
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
mods = [
    'tac.substrates.pact_nerv_distilled_scorer',
    'tac.substrates.pact_nerv_vq',
    'tac.substrates.pact_nerv_bayesian',
    'tac.substrates.pact_nerv_multi_modal',
    'tac.substrates.pact_nerv_diffusion_trajectory',
]
for name in mods:
    __import__(name)
    print(name, 'ok')
PY
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q src/tac/substrates/pact_nerv_distilled_scorer/tests src/tac/substrates/pact_nerv_vq/tests src/tac/substrates/pact_nerv_bayesian/tests src/tac/substrates/pact_nerv_multi_modal/tests src/tac/substrates/pact_nerv_diffusion_trajectory/tests
git check-ignore -v src/tac/substrates/pact_nerv_selector_v2/__pycache__/__init__.cpython-312.pyc src/tac/substrates/pact_nerv_bayesian/__pycache__/__init__.cpython-312.pyc || true
```
