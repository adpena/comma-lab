# FEC6 Selector Sensitivity Fail-Closed And Parent Queue Note - 2026-05-17

## Authority

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_provider_dispatch=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_attempted=false`

No provider dispatch was launched and no lane claim was opened by this
follow-up.

## Parent OMX Markdown Refresh

The operator warned that useful OMX/Claude signal may sit outside
`.omx/research`. I checked the direct parent/state Markdown surfaces:

- `.omx/notepad.md`
- `.omx/release_manifest_v0.2.0-rc1.md`
- `.omx/state/current_focus.md`
- `.omx/state/next_experiments.md`
- `.omx/state/active_dispatches.md`
- `.omx/state/dispatch_queue.md`
- `.omx/state/active_lane_dispatch_claims.md`

The current authority remains `.omx/state/current_focus.md`,
`.omx/state/next_experiments.md`, and
`.omx/state/active_lane_dispatch_claims.md`.

Concrete parent-Markdown finding: `.omx/state/dispatch_queue.md` is ignored
live state and still contains stale April/early-May `READY-NOW` rows when
opened directly. The tracked authority file `.omx/state/current_focus.md`
already classifies it as historical; this ledger preserves the same warning so
the pushed source of truth does not rely on ignored state edits.

## FEC6 Selector Sensitivity Hardening

The untracked FEC6 sensitivity selector WIP had the right intent but the
library-level manifest was not fail-closed enough. It emitted
`score_claim=false`, while some promotion/dispatch false fields were only added
by the CLI wrapper.

Fixes:

- `tac.fec6_selector_discovery_sensitivity_weighted` now emits the full
  false-authority bundle at the core helper layer:
  `promotion_eligible=false`, `rank_or_kill_eligible=false`,
  `ready_for_provider_dispatch=false`, `ready_for_exact_eval_dispatch=false`,
  and `dispatch_attempted=false`.
- Candidate `delta_d_pose`, `delta_d_seg`, and weighted scores must be finite.
  NaN/inf rows are refused before any argmin decision can silently pick or bury
  a candidate.
- The FEC6 selector sensitivity tests now assert the full authority bundle and
  the non-finite refusal behavior.

## Routing Consequence

This is a planning/probe hardening patch, not a score-lowering claim. The
selector remains a diagnostic Rule #6/FEC6 operator-space tool until a
byte-different packet proves runtime consumption and component movement on the
correct axis.

Current P0 routing is unchanged:

1. Rule #6 byte-closed bolt-ons on A1/FEC6.
2. TT5L/L5-v2 side-info effect curve with paired CPU/CUDA custody.
3. FEC6 selector work only with new component rows or a component-moving
   packet operator, not same-runtime byte-only retreads from the current table.

## Verification

Run before commit:

```bash
.venv/bin/python -m pytest src/tac/tests/test_fec6_selector_sensitivity_consumer.py
.venv/bin/python -m ruff check src/tac/fec6_selector_discovery_sensitivity_weighted.py src/tac/tests/test_fec6_selector_sensitivity_consumer.py tools/reweight_fec6_selector_discovery.py
.venv/bin/python -m py_compile src/tac/fec6_selector_discovery_sensitivity_weighted.py src/tac/tests/test_fec6_selector_sensitivity_consumer.py tools/reweight_fec6_selector_discovery.py
git diff --check -- src/tac/fec6_selector_discovery_sensitivity_weighted.py src/tac/tests/test_fec6_selector_sensitivity_consumer.py tools/reweight_fec6_selector_discovery.py .omx/research/fec6_selector_sensitivity_fail_closed_and_parent_queue_note_20260517_codex.md
```
