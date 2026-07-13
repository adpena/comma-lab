# FEED — V9 CGauge event-native ideal config and matched mod19/mod32 A/B

`written_utc: 2026-07-13`

`lane_id: lane_v9_cgauge_ideal_event_eikonal_ab_20260713`

`research_only=false; launch_authority=HELD_operator_GO; score_claim=false; pointer_delta=NONE`

## Canonical graph

1. `DESIGN_COMMITTED` — `.omx/research/v9_cgauge_truly_optimal_design_20260712.md`.
2. `DSL_COMPILE` — `tac.witness_dsl.spec_v9_cgauge.compile_v9_cgauge_ideal_launch_config`;
   named core, mod19, and mod32 programs; **MEASURED-at-build 0 validation violations**.
3. `MANIFEST_SINGLE_OWNER` — compiled DSL argv derives overlapping constants-manifest values;
   the inherited hosc mismatch is repaired from **10.0 manifest / 3.177 argv** to **3.177 = 3.177**
   and guarded fail-closed.
4. `TAU_RUNG_RELAXATION` — persisted `TauAdvanceController`; event/cap proposes exactly one
   geometric rung.
5. `RETENTION_PRIME` — canonical equation
   `eikonal_retention_couples_to_tau_rung_v1` computes
   `lambda_k = lambda_0 + (lambda_N-lambda_0) k/N`; trainer emits the prime receipt before it
   assigns the lower render/loss tau.
6. `RUNG_RETREAT` — the accepted proposal clears the old spike scale, resets AdamW moments when
   armed, and anchors the **DERIVED 14-epoch** beta2-window LR rewarm before lower-tau assignment.
7. `REPAIR_FORCE` — fitted class-pair length sigma and tie-locus are live; lane-nucleus actuates
   the lane band; annulus-plateau actuates chroma and temporal-screw forces. MarginBand is absent
   under sibling ownership.
8. `PRESERVE_STAGE` — after tau and hosc-beta are assigned, save preserved
   `stageOctave<k>` checkpoint with the same persisted controller rung; periodic checkpoint cadence
   remains 25 epochs and all stage checkpoints remain distinct.
9. `ACCEPT_OR_ROLLBACK` — current executable policy is bounded containment: through-R verdict
   trajectory feeds the rung controller and closed-loop erosion fuse; every rung has a complete
   pre/post custody point; a rejected rung stops cleanly and the governed continuation resumes the
   preserved pre-rung checkpoint. **INFERRED limitation:** the trainer does not yet perform an
   autonomous full-facet restore inside the same process; no such capability is claimed.
10. `MUON_GATE` — existing `powerlaw_meat` plus nucleus positive control; tau freezes before Muon.
11. `APPEARANCE_PHASE` — existing phase-advection tail on the formed trunk; dense stored phase
    carrier remains an isolated exact-byte A/B.
12. `FAMILY_BRANCH` — same nodes 2–11 branch only on `mod_dim in {19,32}`; operational output
    directories differ for custody. Decision: revert to mod32 if
    `(dseg19-dseg32)/dseg32 > 0.02` on matched **n600 per-class through-R** verdicts.
13. `RECEIVER_CLOSE` — not executed. Only exact archive bytes, parse-back, deterministic inflate,
    and exact CPU/CUDA rows may move the pointer.

## Triality

- DSL: `src/tac/witness_dsl/spec_v9_cgauge.py`; three named compiled programs; launcher branches
  in `tools/launch_witness_run.py`.
- Equation: `src/tac/canonical_equations/eikonal_retention_tau_rung_20260713.py`; registry event
  `eikonal_retention_couples_to_tau_rung_v1` at the locked canonical equation ledger.
- DAG/consumer: this FEED; live consumer is the level-set trainer; dry-run custody is under
  `experiments/results/v9_cgauge_{ideal_mod19,ideal_mod32,truly_optimal_core}_20260713/`.

## Verdict scope and containment

- **MEASURED source fact:** the dead arm's unified-tau branch bypassed the discrete event resolver,
  so the stage sentinel never fired the end weight.
- **DERIVED law:** retention follows persisted rung `k/N` and is ordered before tau assignment.
- **UNMEASURED:** treatment delta-S, class-wise improvement, pose impact, byte impact, and family
  outcome. The build is means, not an achieved score.
- No training, evaluator, paid dispatch, daemon spawn, archive mutation, or live-run touch occurred.
- Pointer remains unchanged. Contest CPU, contest CUDA, and macOS advisory axes remain separate.

## STORES CONSULTED

Full CLAUDE/AGENTS/operating manual; committed V9 design and #432 config memo; dead-run launch,
run log, and costate shadow; canonical frontier/lane/subagent/equation surfaces; latest Codex/Claude
memo pointers; trainer argparse/control paths; DSL gauge/curriculum/campaign and typed compiler;
tau controller/resume registry; measured edge-weight artifact; governed launcher memory, safe-compile,
and system-admission preflights. The sibling-owned MarginBand factory/equation was read only for the
ownership boundary and was not edited.

