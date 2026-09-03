# ddm_wc3_qbr1_ema_law_cure — cure WC2-F1 (executable EMA law ≠ sealed law) in the QBR1 born fair-form burn, land the STRICT self-protection gate, re-seal, rerun resume identity, and hand MAIN a typed scorer-lane fire order for the six-cell burn (the distortion-corner object-change route)

## MANDATE

Operator 2026-09-03: *"continue with all"* under the standing GO. The QBR1 six-cell burn is the
sealed distortion-corner route (born object, fair-form discriminator) and MAIN's Metal slot idles BY
STATE because no VALID sealed successor exists: WC2 (`ddm_wc2_qbr1_bug_wallclock_realization_audit_20260902.md`,
acd96775ec) issued a BURN-INVALIDATING fire alarm. The authorized config resolves
`ema_decay_run_geometry_v1` in constant-decay seed-fraction mode (5,000 updates, terminal seed
fraction 0.01, decay 0.9990793899844618) but the burn entry point constructs
`qbt.EMA(model, decay=..., warmup=True)` (`experiments/ddm_qbr1_born_fairform_burn_prep.py`
ddbab64a60, lines 496 and 679), whose executable law is `min(d, (1+t)/(10+t))`: declared terminal
coefficient d^5000 = 0.010000000000000278 vs executed 1.838001854879489e-27; the cap is not reached
until update 9,767. Milestones and candidate materialization consume the shadow, so the running cell
measured a different intervention than its seal. WC2 itself ended BLOCKED-LOCAL-TOOLING with its
realization rows unfinished. This is a confound of the DEFAULT-HARMFUL × SILENT × MEASUREMENT-CORRUPTING
signature; CLAUDE.md "Confound self-protection" binds the 3-layer cure.

## SCOPE

1. **Cure (L0):** make the executable EMA law EQUAL the sealed law. The registered LawRef
   (`src/tac/witness_dsl/scope_laws.py` 78f41ed1e7 + `curriculum_dsl.py`) is the authority: when the
   config's EMA mode resolves to constant-decay geometry, construct `EMA(..., warmup=False)`; keep
   `warmup=True` ONLY as an explicit, config-declared ablation. Thread the choice from the DSL-compiled
   config — never a hardcoded flag (config ONLY via typed DSL compile; the lever must be a `Lever`
   factory or an existing one's field, per the triality rule). Stamp `ema_law_executed` +
   `ema_law_sealed` + their closed-form terminal coefficients into run provenance.
2. **L1 runtime alarm:** at EMA construction, compute both closed-form terminal coefficients and emit a
   typed `confound_alarm(ema_law_mismatch)` + HALT if they differ beyond 1e-12 relative.
3. **L2 STRICT gate (two-landing rule):** `check_ema_executable_law_matches_sealed_law` in
   `src/tac/confound_gates.py` — AST scan for `EMA(` constructions whose `warmup` argument is a
   literal while the enclosing module resolves an `ema_decay_*` LawRef; same-line
   `# EMA_WARMUP_ABLATION_OK:<rationale>` waiver (placeholder rationales rejected). Wire into
   `preflight_all()`; land WARN-ONLY, verify live count 0 after the cure, strict-flip in the SAME
   commit batch. 15+ tests. Catalog row via `tools/claim_catalog_number.py claim` + the catalog doc
   pointer (`docs/meta_bug_class_catalog.md`).
4. **Re-seal + resume identity:** regenerate the sealed config under the cured law
   (`tools/make_candidate_seal.py` pattern / the QBR1 seal path in the burn script), then run the
   sealed `resume-smoke` at n=1 pair (the SEALED-BLOCKED-ON-MAIN-SCORER-LANE row's command in
   `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/FIRE_ORDER*.json` — read it; it
   requires cursor, live-state, EMA-state, and archive equality). The resume smoke is scorer-free
   at n=1 — if it needs the local scorer lane, STOP and emit the fire order instead (MAIN's lane).
5. **Finish WC2's unfinished rows** (static timing, realization rows, staged-patch verification) as
   far as local tooling allows; each row typed MEASURED / NOT-MEASURED-WITH-REASON.
6. **Hand-off:** a typed FIRE ORDER for MAIN: exact command for the six-cell burn under the cured
   seal, preconditions (unique live scorer claim, Metal slot free, AP/Vertigo free bytes, source pins
   re-verified incl. the 5,078,017,610-byte `gt_n600.npz`), expected wall-clock from WC2's timing
   rows, and the adjudication command.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO scorer run, NO Metal burn, NO Modal fire from the arm — MAIN owns the
  scorer lane, the Metal slot, and dispatch; do not write lane occupancy into reasoning (#1210 genus).
- Do NOT write to the live burn's run directory (`/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/runs/`)
  — run dirs are sacred; the invalidated cell stays retained as-is, labeled.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes;
  ruff clean; `preflight_all` green; the strict gate's tests green.
- ALWAYS KEEP THE PAYLOAD (resume-smoke checkpoints + archives retained under
  `/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/`; refuse if AP free < 1.5 GiB).
- DETACHED >30-MIN COMPUTE: any step projected >30 min launches ONLY via `.venv/bin/python tools/launch_detached_process.py --output-dir <run_dir> --done-receipt <name> -- <cmd...>` (hand-rolled detaches are guard-blocked); the arm monitors, MAIN harvests.
- Resumability P0 unchanged: the cure must not break `--resume-from`; resume identity is the proof.
- Sister arm `ddm_fpc1_full_pipeline_compress` owns the NEW pipeline's EMA construction; you own
  `ddm_qbr1_*`, `ddm_qbt1_qbflow_trainer.py`'s EMA construction site, `confound_gates.py`, and
  `tac.training.EMA` docs. Do not edit `experiments/semantic_joint_ctxmix_pipeline.py`.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_wc2_qbr1_bug_wallclock_realization_audit_20260902.md` — WC2-F1 (the defect) + the
  BLOCKED-LOCAL-TOOLING rows this arm finishes.
- `spike-guard-median-freeze-deadlock-ep-loss-zero-signature` (memory) + `confound_hunt_synthesis_20260705`
  — the 18-confound hunt; a default that silently changes the measured intervention is the genus.
- `feedback_never_launch_non_resumable_per_stage_checkpoints_20260627` (memory) — EMA shadow saved
  per stage; the cure must keep the shadow semantics for byte-close.
- `ddm_qbr1_born_fairform_burn_prep_20260902.md` — the seal contract (cursor/live/EMA/archive equality)
  the re-seal must re-pass; the six cells and their adjudication command.
- `p0_ema_calibration_20260717` (operator-P0 ledger) — `ema_decay_run_geometry_v1` is the DERIVED law
  (2 anchors); the 0.997 constant is a borrowed constant — the cure must resolve through the LawRef.

## OPTIMAL FORM

- Family exemplar: the confound-gate two-landing pattern, reference `src/tac/confound_gates.py`
  (Catalog #397/#398 `check_no_spike_guard_defaults_to_deadlock_mode` /
  `check_reject_filter_updates_reference_from_accepted_only_has_rearm`; commit pin via
  `git log -1 -- src/tac/confound_gates.py`, record it in the memo) and the canonical EMA
  `tac.training.EMA` (`src/tac/training.py` commit 7039585998, warmup semantics documented at lines
  ~509–526).
- SCOPE reductions: n=1-pair resume smoke (legal). MECHANISM reductions FORBIDDEN: no "just set
  warmup=False" without the DSL-resolved law + alarm + gate (a point-fix ≠ a class-fix).
- **PRIOR-LAW PREDICTION (falsifiable):** under the cured law the n=1 resume smoke passes all four
  equalities (cursor/live/EMA/archive) bit-identically, and the L2 gate's live count is 0 after the
  cure with ≥1 positive-control violation caught in tests. FALSIFIER: any equality fails, or the gate
  cannot catch the original line 496 form — count it plainly.

## DELIVERABLE

`.omx/research/ddm_wc3_qbr1_ema_law_cure_20260903.md` — typed rows: cure diff summary + both
closed-form coefficients before/after; gate catalog #, live count, strict flip; resume-smoke equality
table; WC2 rows finished; the typed six-cell FIRE ORDER for MAIN; RECALL EVIDENCE; NEXT_IF_RESUMED;
LIVE-HYPOTHESES; DEAD-ENDS. Commit via the serializer. Cite `docs/operating_manual_craft_handoff.md`.
End with the own-vehicle frontier line.
