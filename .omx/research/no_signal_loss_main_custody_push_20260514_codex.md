# No-Signal-Loss Main Custody Push - 2026-05-14

## Scope

Operator directive: make `main` the sole source of truth for the current Pact
work, push it to comma-lab, preserve `.omx`/research/memory-adjacent state, and
avoid losing experimental signal.

Repository custody at ledger creation:

- repo: `/Users/adpena/Projects/pact`
- branch: `main`
- remote: `origin git@github.com:adpena/comma-lab.git`
- pre-commit HEAD: `3a63e33945d6b1f2008895b78734648d75031bd2`
- fetched `origin/main`: `99ca7859c1d139602d76cacb62f9a92b1db5a9a5`
- divergence before custody commit: `origin/main...HEAD = 0 behind / 169 ahead`

## Preserved durable surfaces

- All non-ignored tracked and untracked code, tests, tools, submission packets,
  reverse-engineering manifests, and `.omx/research` ledgers are staged for the
  main custody commit.
- The compact C6 MDL-IBPS byte-patch evidence is intentionally force-added even
  though `experiments/results/` is ignored, because it is a byte-closed
  candidate/evidence surface rather than rebuildable scratch:
  - `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/`
  - source C6 smoke archive:
    `a27328ce02211f1c8ee0cfb4318ace29c438a62cf09a42358481d0273a204607`
  - CPU600 Tier-B evidence JSON:
    `a625cfd0d3b2f19a94bac0373d75832d056a477182cd6b223722aca82a41aad3`
  - best single-byte patch candidate archive:
    `2d6416874c6563d6f2ebf9b502c98a45b5b4dfee88f42124dbbaa1910579bb3a`
  - best single-byte patch manifest:
    `2fb87092544d0eae3fb127a00204e94bfb8a896d4a11dec425aa19fa6e4e2ed4`
- The C6 Modal smoke structured custody files are force-added where useful
  (`archive.zip`, `stats.json`, provenance, harvest metadata, worker-head
  ledger, terminal claim). Raw provider logs remain ignored by repo policy and
  are summarized by the committed structured metadata.

## Verification attached to this custody push

Focused green tests before this ledger:

```text
PYTHONPATH=src:upstream .venv/bin/python -m pytest \
  src/tac/tests/test_dynamic_video_adaptation.py \
  src/tac/tests/test_mdl_scorer_conditional_ablation.py \
  src/tac/tests/test_mdl_ablation_tier_c_ibps1.py \
  src/tac/tests/test_build_ibps1_byte_patch_archive.py -q

88 passed in 4.16s
```

Static checks before this ledger:

```text
py_compile: DVAR1 modules, MDL ablation tool, IBPS1 byte-patch builder
git diff --check: full current worktree diff clean after EOF whitespace cleanup
```

## Current evidence classification

- C6 CPU600 byte-patch result is `[contest-CPU]`/local advisory evidence from
  the MDL ablation tool, not a public leaderboard or contest-CUDA score claim.
- The best confirmed single-byte offset was `decoder_blob:34858` with
  `delta_score_components=-0.4263202916051725` on 600 CPU pairs in the local
  tool output. It remains blocked from promotion until exact archive/runtime
  custody is replayed on the appropriate contest axis.
- Dynamic per-video adaptation (`DVAR1`) is intentionally fail-closed:
  `score_claim=false`, `promotion_eligible=false`,
  `ready_for_exact_eval_dispatch=false`, and no scorer load is allowed at
  inflate time.

## Residual risks preserved, not hidden

- The commit review gate blocks the bulk landing because the staged custody
  snapshot contains many newly added or stale-review entities (`455`
  review-tracker violations at commit time). This custody push uses
  `REVIEW_GATE_OVERRIDE=1` only to preserve the no-signal-loss source of truth;
  it is not a promotion, release, or score-readiness certification.
- Exact CUDA has not yet been run for the C6 single-byte patch candidate.
- Broader suite drift was observed outside the focused custody slice:
  `test_rank_dispatches_smoke` currently expects the older substrate count.
- Ignored raw provider logs remain on disk but are not promoted into git by
  policy; structured metadata and exact compact archives are committed instead.
- `main` is ahead of `origin/main` before this custody commit and must be
  pushed after commit for comma-lab to reflect this preserved state.
