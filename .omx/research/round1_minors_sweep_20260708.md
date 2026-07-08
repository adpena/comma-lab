# SEAL v7 ROUND-1 MINORS SWEEP (2026-07-08) — [no-triality]

Role: ROUND-1 MINORS SWEEPER (Opus, surgical). Fixed every finding NOT owned by an in-flight fixer,
per operator standing policy 2026-07-08 ("Fix all issues found each round regardless of severity").
NO launches. Pointer 0.19109982 UNMOVED — every fix here is a control-surface/documentation MEANS.

## STORES CONSULTED
- `.omx/research/t5_crucible/seal_v7_r1_deepmath_20260708.md` (bdb79ff9c) — 4 MINORs (2 MAJORs = TAIL fixer).
- `.omx/research/t5_crucible/seal_v7_r1_confound_20260708.md` (06330c3b4) — MINOR-2, MINOR-3 (MINOR-1 dwell_at_cap = MAJOR-1 fixer, already folded at HEAD).
- `.omx/research/t5_crucible/SYNTHESIS_seal_v7_round1_20260708.md` — R-6 disposition wording; standing-policy append.
- Code read at HEAD: `src/tac/witness_control/tau_advance.py`, `tail_cycles.py` (HEAD blob), `src/tac/witness_autoconfig.py` (β-end region), `src/tac/tests/test_tau_advance_self_paced.py`.
- Coordination: verified in-flight fixer footprints via `git diff HEAD` before EVERY hot-file edit; all hot-file commits via serializer `--patch-file` (temp index from HEAD, working tree ignored → no sibling-hunk leak, no clobber of uncommitted fixer work).

## PER-FINDING FIX TABLE
| Finding | Fix | Commit |
|---|---|---|
| deepmath MINOR-3 (flat per-octave cap ignores critical-slowing) | `derive_octave_max_dwell` docstring: KEEP flat cap deliberately (a rung-scaled growth law would be a bare guessed constant / new unmeasured knob — value-provenance violation); the LOUD `cap_fired_before_event` S5 row is the honest data-driven re-calibration path. | 99ce07e44 |
| deepmath MINOR-4 (event-mode LR is a step function) | `lr_anneal_fraction` docstring: explicit BEHAVIOR-CHANGE note — LR jumps at each octave advance (vs incumbent smooth cosine), can transient Muon moments; intended, and why run-1 is CLOCK mode. | 99ce07e44 |
| deepmath MINOR-5 (meat_floor & stop_marginal_s both 1e-4) | `TAU_OCTAVE_MEAT_FLOOR` comment: UNITS note — d_seg units here vs S/ep on the TAIL; shared literal is coincidental, tune/derive independently. (TAIL `stop_marginal_s` side = MAJOR-1 fixer's ν·forfeit derivation, which breaks the coincidence.) | 99ce07e44 |
| deepmath MINOR-6 (len off-by-one: `_cycle_exit` inclusive vs `_cycle_net_marginal` interval) | tail_cycles.py comments at both sites: deliberate convention split (inclusive COUNT for floor/dwell threshold; INTERVAL count for per-epoch RATE). Coordinated patch-file on TAIL fixer's hot file (my hunks in the unchanged-context region 212–243, between their hunks). | 12c1ad8f2 |
| confound MINOR-2 (sensor grades EMA shadow, not live) | Named constant `SENSOR_GRADED_STATE="ema_shadow"` + explicit code note (conservative-late fire is the INTENDED bias) in the module constant block + `ingest` docstring; `graded_state` telemetry field stamped on the event advance row (auditable, never read back into training); +1 test. | 99ce07e44 |
| confound MINOR-3 (octave dwell `--verdict-every` coupling) | Module docstring RESUME-INVARIANT note: `--verdict-every` is part of the τ-advance determinism contract; a resume with a different cadence would not reproduce the τ trajectory. Invariance already covered by `test_resume_mid_octave_reproduces_identical_subsequent_tau_sequence` (cited). | 99ce07e44 |
| structure R-6 (β-end 10 vs blind 1→4) | witness_autoconfig.py KEEP-with-provenance comment at the `hosc_beta_end` pin: blind 1→4 SUPERSEDED (pre-anneal-fix era value); v6-sealed MEASURED anchor = control β-trajectory (β(726)≈3.18) that end=10.0 reproduces at shared den; annealed-β divergence evidence makes it an anneal ENDPOINT (not a fixed β). Comment-only, no behavior change; region clear of launch-path fixer hunks. | 3c827c1e1 |

Verification: ruff -F clean on all edited surfaces; `test_tau_advance_self_paced.py` 25 passed (24 + 1 new); review_tracker 2 passes per file; every commit post-verified (only my files/hunks at HEAD, no fixer-work clobber — TAIL fixer's 135 uncommitted insertions on tail_cycles.py preserved).

## NOT SKIPPED, NOTED
- deepmath MAJOR-1/MAJOR-2 → TAIL fixer (a45f68…); confound MINOR-1 dwell_at_cap → already folded at HEAD by the persistence/MAJOR-1 fixer. Not touched.
- MINOR-5 TAIL side (`stop_marginal_s`): my fix is the tau_advance units note; the numeric fix (1e-4 → derived ν·forfeit) is the TAIL MAJOR-1 fixer's surface — coordinated, not duplicated.

## RESIDUAL COORDINATION RISK (named, not a blocker)
MINOR-6 landed via `--patch-file` (my 8 lines at HEAD, NOT in the TAIL fixer's on-disk working copy). If a subsequent WHOLE-FILE tail_cycles.py commit lands from a working tree lacking my lines, it could revert MINOR-6. Mitigation: the serializer refuses whole-file on high-risk files without `--expected-content-sha256` + post-commit rc=7 clobber detection; the TAIL fixer follows the same patch-file discipline (their MAJOR hunks don't touch 212–243). If reverted, re-apply the same MINOR-6 hunk against latest HEAD. Nothing genuinely unfixable-now.

## STANDING-POLICY TRAIL
Appended the operator fix-all-severities-before-next-round policy to `SYNTHESIS_seal_v7_round1_20260708.md`
(append-only section): from round 2 onward all-severity findings are fixed before the next round convenes;
the clean-pass counter still only counts zero-findings rounds; zero-unfixed-findings is now a convene precondition.
