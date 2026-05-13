# PR95++ Smoke Contract Hardening - Codex

Date: 2026-05-13
Lane: `lane_pr95_meta_stack_of_stacks_enhanced_curriculum_20260513`
Evidence axis: `[macOS-CPU advisory]` local smoke; `[contest-CUDA]` required on Modal smoke

## Finding

The prior PR95++ Modal smoke failed before training with rc=13 because a
worker-side sentinel was outside the Modal mount set. Current HEAD already
contains Catalog #201's sentinel mount-set filter and strict preflight guard.

Fresh-eyes review then found the next blocker: the Catalog #167 smoke wrapper
requires a current-run `auth_eval_*.json` with a valid `contest_cuda` score
claim, while the PR95++ trainer emitted only a research manifest and placeholder
archive. A relaunch would likely get past rc=13 and then fail the smoke gate for
missing auth evidence.

## Landing

Patch set:

- `experiments/train_substrate_pr101_lc_v2_clone_enhanced_curriculum.py`
  now emits a valid PRC1 smoke archive instead of a JSON placeholder when the
  short smoke run hits the PR101 negzig non-bijection. The fallback is a typed
  zero-state PRC1 archive, explicitly `score_claim=false`, but contest-inflatable.
- The same trainer vendors the PR101 LC v2 clone runtime package under
  `runtime/src` so `contest_auth_eval.py` exercises the shipped parser,
  decoder, and inflate runtime rather than mutable repo source.
- CUDA smoke runs `experiments/contest_auth_eval.py --device cuda` and refuses
  silent success unless `parse_auth_eval_score_claim(..., contest_cuda)` passes.
- `scripts/remote_lane_substrate_pr101_lc_v2_clone_enhanced_curriculum.sh`
  now forwards `PR95PLUS_SMOKE_EPOCHS` and logs the auth-eval artifact path.
- The recipe declares `PR95PLUS_SMOKE_EPOCHS: "100"` so the smoke-before-full
  helper threads a real epoch budget instead of only changing the cost-band row.

## Local Evidence

Commands:

```bash
.venv/bin/python -m pytest \
  src/tac/substrates/pr101_lc_v2_clone/tests/test_curriculum_enhanced.py \
  src/tac/tests/test_run_modal_smoke_before_full.py \
  src/tac/tests/test_operator_authorize_scripts.py::test_smoke_before_full_wrappers_dry_run_without_smoke_flags \
  -q
```

Result: `89 passed`.

```bash
.venv/bin/python -m pytest \
  src/tac/substrates/pr101_lc_v2_clone/tests/test_pr101_lc_v2_clone_roundtrip.py \
  src/tac/substrates/pr101_lc_v2_clone/tests/test_curriculum_enhanced.py \
  -q
```

Result: `69 passed`.

```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
  --recipe substrate_pr101_lc_v2_clone_enhanced_curriculum_modal_a100_dispatch \
  --dry-run
```

Result: dry-run now reports `epoch_env_var=PR95PLUS_SMOKE_EPOCHS`.

```bash
.venv/bin/python experiments/train_substrate_pr101_lc_v2_clone_enhanced_curriculum.py \
  --curriculum pr95_enhanced \
  --video-path upstream/videos/0.mkv \
  --output-dir experiments/results/pr95plus_local_repro_20260513_codex_patch2 \
  --device cpu --smoke --smoke-epochs 9 --max-pairs 2 --batch-size 1
```

Result: archive emitted with SHA-256
`94bf21731b9617cc6848e5fc75ff3c8d1489d0677e91f2bb40fb02efaa2919c3`,
`720` bytes, `smoke_archive_mode=zero_state_valid_prc1`.

## Status

This is not a score claim and not promotion evidence. It fixes the smoke
contract so the next Modal T4 smoke can produce a real current-run
`contest_auth_eval_cuda.json` or fail closed with a meaningful runtime/scorer
diagnosis.

Next action after preflight/commit: relaunch the smoke-only PR95++ Modal T4
anchor from clean HEAD. Full A100 remains blocked by the recipe's
`smoke_only: true` until full training/export is implemented.
