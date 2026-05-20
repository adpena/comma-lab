# xhigh Adversarial Review: Current Landings And Dirty Worktree Risk

UTC: 2026-05-20T14:20:23Z
Author: Codex
Scope: senior adversarial review of current landing activity, dirty worktree risk, score-authority surfaces, PR #110 freeze posture, and sister-subagent ownership.
Write scope honored: this memo only. No code, docs, state, reports, PR artifacts, or ledgers were edited.

## Executive Verdict

**DO NOT bulk commit the current tree.** The dirty state currently spans multiple unrelated active ownership groups: staged Lane Omega cleanup, unstaged PR110/report artifacts, modified Wave-3 per-axis consumer conversions, untracked Blahut-Arimoto and domain-prior packages, PR110 evidence-pack files including `archive.zip`, and live state rows. A single commit would risk absorbing sister-subagent work and false authority state.

`git diff --check` and `git diff --cached --check` were clean, but whitespace cleanliness is not the relevant blocker. The relevant blocker is ownership, authority, and state contamination.

## Severity Findings

### P0 - Canonical equation state is polluted by test/runtime fixtures

`.omx/state/canonical_equations_registry.jsonl` has new dirty rows from `domain_prior_consumer` containing dummy archive SHA `aaaaaaaa...`, `call_id` values `test_call` / `domain_prior_consumer_update`, and `measurement_utc: "1970-01-01T00:00:00Z"`.

Evidence:
- `.omx/state/canonical_equations_registry.jsonl:16-17`
- `src/tac/cathedral_consumers/domain_prior_consumer/__init__.py:344-350` calls `update_equation_with_empirical_anchor(...)` against the live canonical helper.
- `src/tac/cathedral_consumers/domain_prior_consumer/tests/test_consumer.py` exercises update paths with synthetic anchors.

Risk: false continual-learning / canonical-equation authority. Do not commit this state as-is. The domain-prior tests need a temp registry / monkeypatch, or the intended registry registration must be separated from fixture anchors.

### P0 - Current dirty tree crosses active sister-subagent ownership

Current `git status` has at least these disjoint ownership groups:
- staged deletions: `src/tac/bit_allocator.py`, `src/tac/tests/test_bit_allocator.py`
- modified PR/report artifacts: `reports/latest.md`, PR101/FEC6 `README.md`, `archive_manifest.json`
- modified five cathedral consumers for per-axis emission
- untracked `src/tac/blahut_arimoto/**`
- untracked `src/tac/cathedral_consumers/domain_prior_consumer/**`
- untracked `.omx/research/pr110_final_evidence_pack_20260520T141144Z_codex/**`, including `archive.zip`

Evidence:
- `.omx/state/subagent_progress.jsonl:1948-1956` owns the Lane Omega forensic fix.
- `.omx/state/subagent_progress.jsonl:1949-1952` owns the five-consumer per-axis conversion.
- `.omx/state/subagent_progress.jsonl:1953` owns Blahut-Arimoto.
- `.omx/state/subagent_progress.jsonl:1955` owns domain-prior consumer.

Risk: Catalog #314-style absorption of sister work. Commit only isolated, reviewed groups.

### P1 - `reports/latest.md` has pointer drift and duplicate top-N anchors

`reports/latest.md` claims it is current from `canonical_frontier_pointer_20260520T115711Z`, has `last_refreshed_head: local_dirty_worktree`, and says the pointer was refreshed at `2026-05-20T11:57:11Z`. The canonical pointer file is now refreshed at `2026-05-20T13:47:03.952511+00:00`, and the report's top-5 lists include duplicate anchors.

Evidence:
- `reports/latest.md:3`, `reports/latest.md:7`, `reports/latest.md:12-14`
- `reports/latest.md:50-62` duplicates CPU/CUDA top rows.
- `.omx/state/canonical_frontier_pointer.json:3`
- `.omx/research/frontier_execution_queue_pr110_postreview_20260520T141254Z_codex.md:12-17`

Risk: canonical-pointer drift and report-as-authority confusion. Use the canonical pointer JSON / scan helper as authority until the report is regenerated cleanly from current state.

### P1 - PR #110 freeze boundary is intact locally, but fix recommendations now need freeze-break routing

The freeze guard forbids PR body edits, PR comments, PR branch mutation, and release asset replacement without a new `pr110_freeze_break_<utc>_codex.md`. No such freeze-break memo exists. Several PR110 review memos recommend public-facing fixes, including report custody, public transitive-doc exposure, source-of-truth wording, and release-name/frontier wording.

Evidence:
- `.omx/research/pr110_live_freeze_guard_20260520T141520Z_codex.md:26-38`, `:50-55`
- `.omx/research/pr110_transitive_doc_audit_20260520T141423Z_codex.md:23-62`
- `.omx/research/pr110_local_artifact_sync_findings_20260520T141408Z_codex.md:193-226`
- `find .omx/research -name 'pr110_freeze_break_*'` returned no files.

Risk: accidental PR #110 freeze violation if someone acts on the recommendations directly. Any public PR/release/doc-linked fix needs an explicit freeze-break memo first.

### P1 - PR110 local artifacts still contain score-authority ambiguity

The local manifest has `score_claim: false`, `promotion_eligible: false`, and `ready_for_submission: true` in the same object while also carrying explicit `[contest-CPU]` and `[contest-CUDA]` score observations. The hosted release manifest calls the packet a generic "frontier submission" even though it is CPU-frontier and not CUDA-frontier.

Evidence:
- `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive_manifest.json:8-11`, `:150-155`
- `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/pre_submission_compliance.hosted_archive_manifest.json:6-8`
- `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/pre_submission_compliance.competitive_or_innovative_statement.txt:1`

Risk: false-score-authority ambiguity. If these files are committed, they should either clearly scope `score_claim=false` to an internal queue or add an explicit public-PR score-claim field with axis labels.

### P1 - Five per-axis consumer conversions need targeted tests before commit

The five modified consumers now emit `predicted_axis_decomposition`, but the dirty tree does not include corresponding dedicated tests for those files. There are also sign/scale ambiguity risks:
- `per_pair_gradient_clustering_consumer` names `per_cluster_archive_bytes_saved` but writes it as a positive `predicted_archive_bytes_delta`; if it is truly bytes saved, sign should likely be negative.
- `per_segnet_class_chroma_consumer` comments "100*sum(...)" while `score_composition` already multiplies `predicted_d_seg_delta` by `100.0`.

Evidence:
- `src/tac/cathedral_consumers/per_pair_gradient_clustering_consumer/__init__.py:104-145`
- `src/tac/cathedral_consumers/per_segnet_class_chroma_consumer/__init__.py:157-169`
- `src/tac/score_composition/__init__.py:128`, `:349`
- `src/tac/cathedral/consumer_contract.py:148-180`

Required before commit: focused tests for `AxisDecomposition.from_dict`, provenance shape, non-promotable markers, sign convention, scale convention, and ranker composition for each converted consumer.

### P1 - Lane Omega cleanup is staged but should be isolated

HEAD `d72525543` added `src/tac/bit_allocator/lane_omega.py` and `src/tac/bit_allocator/tests/test_lane_omega.py`, but the legacy `src/tac/bit_allocator.py` and `src/tac/tests/test_bit_allocator.py` are still tracked in HEAD and staged for deletion now.

Evidence:
- `git diff --cached --name-status` shows only those two staged deletions.
- `src/tac/bit_allocator/__init__.py:123-152` re-exports the legacy API from `lane_omega`.
- `src/tac/bit_allocator/lane_omega.py:26-41` records the rename rationale.

Risk: incomplete cleanup if not committed, or accidental coupling if committed with unrelated dirty files. This should be its own commit after allocator/import tests.

### P2 - PR110 evidence pack includes raw rebuildable artifacts

The untracked PR110 evidence pack is about 500 KB and includes `archive.zip` plus runtime snapshots. `archive.zip` is not ignored by `.gitignore`.

Evidence:
- `.omx/research/pr110_final_evidence_pack_20260520T141144Z_codex/archive.zip`
- `git check-ignore` did not report that archive as ignored.

Risk: raw/rebuildable public artifact committed into `.omx/research` instead of a small manifest. Commit only small evidence summaries/manifests unless policy explicitly approves the archive.

## Commit Grouping

1. Lane Omega cleanup only: staged deletions for `src/tac/bit_allocator.py` and `src/tac/tests/test_bit_allocator.py`.
2. Five per-axis consumer conversions plus their focused tests only.
3. Domain-prior consumer package only, after preventing live registry mutation.
4. Blahut-Arimoto package only, with any intended canonical-equation registry row separated from test fixtures.
5. PR110/report docs only, excluding raw evidence-pack archive/runtime snapshots unless explicitly approved.

## Test Gates Before Commit

- Lane Omega: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/bit_allocator/tests/test_lane_omega.py src/tac/bit_allocator/tests/test_per_byte_per_class_per_axis_pareto_dual.py src/tac/tests/test_omega_export_load.py src/tac/tests/test_remote_lane_omega_script.py -p no:cacheprovider`
- Five per-axis consumers: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_master_gradient_exploits_end_to_end.py src/tac/tests/test_score_composition.py src/tac/tests/test_check_356_per_axis_decomposition_provenance.py -p no:cacheprovider`
- Domain prior: same dedicated consumer tests, but only after sandboxing or monkeypatching canonical-equation writes.
- Blahut-Arimoto: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/blahut_arimoto/tests/test_blahut_arimoto.py -p no:cacheprovider`, plus a registry-write audit if it appends to `.omx/state/canonical_equations_registry.jsonl`.

## Bottom Line

The active landings are not blocked by one code bug; they are blocked by separation discipline. The highest-risk concrete defect is live canonical state contaminated with synthetic anchors. The highest-risk process defect is a bulk commit that merges staged cleanup, partner source work, PR110 public-surface artifacts, and raw evidence files into one history unit.
