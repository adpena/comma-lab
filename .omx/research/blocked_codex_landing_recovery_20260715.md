# Blocked-codex-landing recovery — 2026-07-15

**Arm:** `blocked_codex_landing_20260715`. **Charter:** recover + land the 3 dead codex arms named by
the signal-loss sweep (`signal_loss_coherence_pass_20260715.md`, respawn-queue rows 1–3) whose work
finished but could not commit (sandbox read-only `.git`).
**Pointer:** 0.19108 submittable / 0.18804 bank — **UNMOVED** (apparatus/means; no score claim).

## Headline: all three arms were ALREADY LANDED — respawn-queue rows 1–3 are STALE

The sweep report's respawn queue was written from the arms' `blocked` checkpoint rows without
verifying against main. Line-level verification at recovery time shows every arm's finished output
is on main, landed 07-12/07-13 (before the sweep):

| arm | located at | reviewed verdict | disposition | tests run |
|---|---|---|---|---|
| `codex_frozen_segnet_gradient_p0_20260712` (20 files) | all 20 TRACKED on main: 16 code files landed `d880211c96` ("land throughput vehicle #447+#449 … recovered+cleaned+reviewed by codex A1"), 3 memos `2378a1f8ce`, profiler spec `82ae10867b`; working tree clean, no divergent copy remains | ALREADY-LANDED, superseded-in-place (main evolved further: `fbaab5cb76` YOPO provider, `2400d39571`, `7263015bd1`) | `reviewed_committed --commit d880211c96` | 153 passed (elm_inr_head_solve, elm_head_seed_policy, segnet_gradient_replacement, scorer_gradient_policy, profile_segnet_blocks, both canonical-eq suites) |
| `codex_micro_batch_v9_unlock_20260712` | exact-hunk patch `/private/tmp/micro_batch_v9_unlock.patch` intact, sha256 `65807b2f…` matches checkpoint; all 26 patch paths present on main | FULLY SUBSUMED: trainer hunks 115/115 distinctive lines present in `train_levelset_witness_realized_through_R_mlx.py`; loss/probe/levers/DSL/eq files via `d880211c96`; memos via `2378a1f8ce`; 19 residual "missing" lines all verified as reviewed-refactor equivalents (e.g. `mx.stack([…])` vs `(…)`, reordered `__slots__`, reworded SHARPENER docstring). FreSh-factory review blockers resolved by later landings | `reviewed_committed --commit d880211c96` | 267 passed / 11 failed — failures PRE-EXISTING (below) |
| `codex-sfess-cached-replay-20260712` (= delegation `sfess_396`) | all 19 files TRACKED, landed `141fcf23aa` ("harvest sfess_396, NO-GO instance-scoped"); authoritative receipt `experiments/results/sfess_cached_replay_ugc64_20260712T214520Z/measurement_receipt.json` sha256 `aa296c61…` VERIFIED byte-identical to the arm's recorded value (gitignored results dir, memo carries custody); sfess provider hunks in `segnet_gradient_replacement.py` + eq registry (5 rows) + DAG (6 refs) present | ALREADY-LANDED; precise disposition supersedes the generic `seed-baseline` close | `reviewed_committed --commit 141fcf23aa` | sfess suites green within the 153-passed run |

No serializer landing of arm content was needed or performed — zero unlanded bytes remained; the
only mutations are ledger rows (gitignored live state) + this note.

## Pre-existing main finding (recorded, NOT fixed — live-arm domain)

`test_levelset_micro_batch_loss.py` (10) + `test_micro_batch_bit_identity_probe.py` (1): parity
receipts fail on `loss_abs ≈ 0.0039 > loss_abs_tolerance=1e-4` while `loss_rel ≈ 4.5e-8` — relative
parity holds; the ABSOLUTE tolerance is miscalibrated against the post-D15 loss scale
(`bd6219a0a3` routed `--logit-adjust-loss-tau`/`--seg-form-unify-tau` into the micro-batch twin).
Owned by the live #509/#507 loss-internals domain (entangled files); recorded here + in the
disposition rather than fixed, per the no-trample rule.

## State mutations (all append-only)

- 3 × `reviewed_committed` rows in `.omx/state/codex_landing_ledger.jsonl`
  (labels `codex_frozen_segnet_gradient_p0_20260712`, `codex_micro_batch_v9_unlock_20260712`, `sfess_396`).
- 3 × `complete` closure rows in `.omx/state/subagent_progress.jsonl` for the blocked arm ids —
  future respawn sweeps must NOT re-queue these three.
- This note (APPEND-ONLY correction; the sweep report itself is unmutated per Catalog #110/#113).
