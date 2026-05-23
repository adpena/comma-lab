# Codex Findings - Inverse Scorer Path Custody Hardening

UTC: 2026-05-23T19:20:15Z
Lane: `lane_inverse_scorer_path_custody_20260523`

## Finding

IAS1 candidate and parity manifests needed an explicit path-custody guard so a
future proof could not resolve candidate, source, or manifest references through
parent traversal or symlink aliases. This matters because parity proof rows can
clear `candidate_inflate_output_parity_missing`; their archive and descriptor
bindings must not be silently redirected.

## Fix

- Candidate archive custody now rejects relative manifest paths containing
  parent traversal.
- Candidate archive custody now rejects symlink archive paths.
- Runtime-adapter candidate-manifest references and candidate-archive references
  resolve only through repo-local or manifest-sibling safe children.
- Inflate parity probes record parent-traversal blockers for manifest archive
  paths and runtime parity fallback resolution refuses unsafe parent traversal.
- Inflate parity from archives now refuses to delete an existing work directory,
  refuses symlinked runtimes and `inflate.sh`, blocks SHA-mismatched archives
  before execution, rejects unsafe ZIP members, and only treats `0.mkv` as a
  full-frame file-list claim.
- Materializer output/template paths now reject symlinks before overwrite or
  archive construction.

## Verification

- `src/tac/tests/test_inverse_scorer_cell_materializer.py`: 38 passed.
- Integrated focused slice
  `src/tac/tests/test_inverse_scorer_cell_materializer.py`
  `src/tac/tests/test_artifact_retention.py`
  `src/tac/tests/test_optimizer_exact_readiness.py`
  `src/tac/tests/test_exact_dispatch_authority.py`
  `src/tac/tests/test_byte_shaving_campaign_queue.py`: 127 passed.
- `git diff --check`: passed.
- `compileall` for `inverse_scorer_cell_inflate_parity.py` and
  `inverse_scorer_cell_materializer.py`: passed.

## Authority

This is a custody hardening guard only. It does not claim score, promotion,
rank, kill, or exact-eval dispatch readiness.
