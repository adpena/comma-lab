# Consolidation quarantine: implementation-orphaned tests (2026-07-14)

`research_only=true`  
`verdict_scope=PILE_INTERNAL_ORPHAN_TESTS_ONLY`  
`frontier_pointer=UNCHANGED`

## Disposition

The post-pile full-suite collection gate found two tests that imported modules which were never
committed and which the consolidation review did not promote:

- `src/tac/tests/test_log_sobolev_tau_anneal_20260714.py`, introduced by `855735f216`, imported
  missing `tac.training_curriculum.log_sobolev_tau_anneal_20260714`.
- `src/tac/tests/test_metric_unification_synthesis_20260714.py`, introduced by `ad7e118e11`,
  imported missing `tac.research.metric_unification_synthesis_20260714`.

Both complete test files were adversarially read before quarantine. Git history preserves their
exact bytes at the cited commits. No implementation blob for either imported module exists in any
reachable ref, and neither module has another repository consumer. Restoring only these tests would
leave pytest collection broken; inventing replacement implementations during consolidation would
silently promote two unreviewed research designs. The tests are therefore removed from the live test
suite and retained as historical hypotheses through their source commits and the existing research
artifacts.

## Scope

This is not a negative verdict on log-Sobolev annealing, DE/LSI cross-checks, metric unification, or
their representation families. Reintroduction requires a dependency-closed implementation plus its
tests in one reviewed landing. No score, evaluator, launch, or promotion claim follows from this
quarantine.
