# Codex Findings: Dreamer Runtime Bridge Stack Contract

UTC: 2026-05-31T03:57:06Z

## Scope

Adversarial follow-up on the MLX/archive bridge migration and the validated
predictive-coding stack-of-stacks contract. Focus: orphaned DreamerV3 RSSM
MLX signal, archive-bound receiver custody, provenance-clean stack execution,
entropy-stage classification, and false-authority preservation.

## Finding

DreamerV3 RSSM had a real MLX archive grammar and inflate runtime, and partner
work had moved it into real-Hinton signal, but the stack-of-stacks contract
still marked Dreamer as archive-bound bridge missing. That left the bounded
runner seeing Dreamer as a validated member but not as an executable
archive-bound candidate source.

## Fix Landed

- Added `tac.substrates.dreamer_v3_rssm.archive_candidate`.
- The new bridge exports RSSMC1 `0.bin`, emits a deterministic contest-shaped
  `archive.zip`, vendors the decode-only runtime, runs generated `inflate.sh`,
  and writes the shared archive-bound candidate adapter package.
- Wired the public Dreamer package exports.
- Updated the predictive-coding stack contract so Dreamer now has a concrete
  archive-bound bridge entrypoint. Z8 remains the only bridge-missing member.
- Extended the archive-bound candidate classifier so Dreamer/RSSM/Z8 rows keep
  predictive-coding substrate tags and before-coder entropy position.

## Authority Discipline

This remains MLX-local and receiver-proofed but score-authority false. The
bridge proves runtime consumption and archive custody; it does not claim score,
promote, rank/kill, spend budget, or dispatch exact eval. Exact CPU/CUDA
authority still requires the normal preclaim and auth-axis handoff.

## Verification

- `.venv/bin/python -m ruff check` on touched TAC files.
- `PYTHONPATH=.:src .venv/bin/pytest src/tac/tests/test_archive_bound_runtime_bridge_remaining_mlx_emitters.py::test_dreamer_export_emits_shared_archive_bound_package_fail_closed src/tac/tests/test_predictive_coding_stack_of_stacks.py src/tac/tests/test_repair_family_materializers.py::test_archive_bound_candidate_contract_classifies_entropy_substrates -q`

## Remaining Work

The next bridge gap is Z8. It has canonical archive bytes and a real inflate
runtime, but a full receiver proof can emit the contest-size raw output, so the
right next patch is a bounded Z8 proof mode that still exercises decode while
preserving the exact contest runtime for promotion.
