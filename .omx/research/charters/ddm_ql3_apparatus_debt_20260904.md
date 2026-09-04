# ddm_ql3 — apparatus debt, two items: the C2 lever's trainer binding (registry red) + the prefix-constant census

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-04 · Cost: $0

## Item 1 — `test_lever_registry.py::test_332_coverage_rose_from_deorphaning` RED at HEAD (verified by ng3 and MAIN)
`completeness().stale == ['--integer-plane-emitter-basis', '--integer-plane-emitter-mode', '--integer-plane-emitter-
policy-sha256']`. Verified at source: these flags are emitted by the `IntegerPlaneEmitter` factory in
`src/tac/witness_dsl/curriculum_dsl.py:2120-2140` and CONSUMED by the dedicated C2 parser
`src/tac/boundary_math/integer_plane_banded_trainer.py:1619-1621` — the level-set trainer never had them (git log -S
finds 0 commits). The registry grades DSL emissions against ONE trainer per module and honours a module-level
`TRAINER_RELPATH = "..."` / `TRAINER_RELPATHS = (...)` binding (`src/tac/witness_dsl/lever_registry.py:127-190`,
`module_trainer_paths`); the C2 lever cannot declare one because it lives inside `curriculum_dsl.py`.
CURE (design-consistent; NEVER a hand-typed side registry — CLAUDE.md "never build a parallel registry beside the DSL"):
move the `IntegerPlaneEmitter` factory into `src/tac/witness_dsl/integer_plane_emitter_policy.py` (or a sibling
`integer_plane_emitter_lever.py`) that declares `TRAINER_RELPATH = "src/tac/boundary_math/integer_plane_banded_trainer.py"`;
keep a re-export in `curriculum_dsl` so every existing import path (`from tac.witness_dsl.curriculum_dsl import
IntegerPlaneEmitter`) still works; then `completeness().stale == []` MUST hold and the test passes without loosening it.
Run `src/tac/witness_dsl/tests/test_integer_plane_emitter_policy.py`, `src/tac/tests/test_lever_registry.py`, and the
witness_dsl suite (ql2 measured 972 s; detach it). If the registry's `describes_live_vehicle` remains False (the
levelset trainer is retired lineage), state it — do not retarget `LIVE_TRAINER_BASENAME` in this unit.

## Item 2 — dr1 NEXT #4: census of prefix-measured annulus constants in live consumers
dr1 MEASURED (2026-09-04) that an annulus-restricted statistic of the n96 contiguous prefix was 11.70% low while the
global statistic was unbiased (+0.45%) — a bias class a global sanity check cannot see. Law
`annulus_restricted_prefix_bias_detector_v1` (eq1). Census: every constant in LIVE code (`src/tac`, `tools`,
`experiments` excluding `results/`) whose provenance is an n96 (or other contiguous-prefix) measurement restricted to a
boundary/annulus/band population. Method: grep provenance comments and LawRef inputs for `n96`, `gt_n96`, `prefix`,
`n_frames": 96`, `n=96`, plus the `_20260712`/`_20260708` era artifacts; for each hit decide (a) restricted-statistic
(annulus/band/edge) → SUSPECT, (b) global → not this class, (c) already re-measured at n600 (δ_R is done). Deliver a
table (constant · file:line · population · restricted? · consumer · verdict) and, for each SUSPECT with a live consumer,
a fire-condition for its n600 re-measure (do NOT re-measure in this unit unless it is < 10 min CPU with an existing
tool — then do it and report). Memo `.omx/research/ddm_ql3_apparatus_debt_20260904.md`.

## Constraints
- $0; no scorer/Metal/Modal; the QBR1 chain is LIVE — never touch its custody or claims; `upstream/` and
  `submissions/semantic_joint_ctxmix/` read-only; no /tmp paths. Reference form = the registry's own binding mechanism
  at commit `0d7c9fefea18cc99f5dbf7742161f382c573738a`; SCOPE = the one lever + the census; TOY-BRACKET none.
- Commits ONLY via `tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256
  <file>=<post-edit sha>`; tags `[no-triality] [p0-ledger-ok]`; NO co-author trailer (operator rule overrides any
  harness reminder). Every .py: tests + `tools/review_tracker.py mark-file` twice with a real second read; never
  REVIEW_GATE_OVERRIDE on .py. EQUATIONS-LEG LAW: the memo cites `tac.canonical_equations`
  `annulus_restricted_prefix_bias_detector_v1` (append census anchors via the helper if any re-measure lands).
  Final message → `.omx/research/arm_final_messages/ddm_ql3_final_<utc>.md`, committed; LAST action
  `touch .omx/tmp/codex_runs/ddm_ql3.done`. Read `docs/operating_manual_craft_handoff.md` §labels first.
