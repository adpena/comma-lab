# ddm_pm2 — the day's instrument laws become STRUCTURAL GATES (two-landing rule): pose-base magnitude gate · encoder-must-be-the-pointer's-coder assertion · ledger-persisted-before-dump preflight · screening-law declaration at charter lint (charter, 2026-09-09)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm (gpt-6-astra, high) · Spawned by MAIN 2026-09-09 (operator: "Codex should be available too"). Sources: gs3 Addendum 18 §IV item 4 (`ae6a0b837`); memories `pose_base_must_be_measured_on_the_pointers_own_configuration_on_the_arms_instrument_20260909` (a base measured without `--overlay` was 500× off and was caught by MAGNITUDE, not by a gate), `renderer_int4_code_grid_smallest_action_breaks_240_cells_20260909` (screening law: a subset screen is valid only for pair-confined actuators; rw1's "repair" retracted at n600), rp1's ledger loss (`np.concatenate([])` on an empty accepted set destroyed a ledger after the dumps; the ledger must persist first), and the stale-coder trap MAIN found 2026-09-09 22:30Z: `experiments/ddm_sj1_successor_price.sh` pins ENCODER to the cl2 HPAC receiver copy while the live pointer's tail is tc1's mixer — a number from it would price the wrong coder with no refusal. CLAUDE.md "Bugs must be permanently fixed AND self-protected against" + "Confound self-protection" (L1 alarm + L2 gate + L3 verdict clearance). Axes: apparatus; `score_claim=false`.

## MANDATE
Four confounds of the DEFAULT-HARMFUL × SILENT × MEASUREMENT-CORRUPTING kind surfaced this day and each was caught by a person's eye. Make each loud and refusable:
1. **Pose-base magnitude gate.** In `experiments/ddm_sj1_joint_admission.py` (`load_pose_instrument`, `pose --tag base`) and every sibling that measures a base (rp1's `pose_base`, fe1's): after measuring, compare to the live pointer row's recorded d_pose (`.omx/state/canonical_frontier_pointer.json`, the promoted receipt's component); refuse (typed `confound_alarm`, rc≠0, no artifact written as "base") when the ratio leaves [1/3, 3], with the message naming the likely cause (missing `--overlay`, wrong tree). Waiver only by explicit flag with a non-placeholder rationale.
2. **Encoder-is-the-pointer's-coder assertion.** Any pricing entry point that takes a `--runtime-root`/ENCODER (jg2 tail re-encode, the sj1 price scripts, tc1/sm1/rc3 codecs' pricing CLIs): assert the encoder tree's runtime digest (`tac.candidate_seal.measure_runtime_digest`) equals the live pointer tree's, or that the caller passed `--coder-differs-because <rationale>`; the control stage's byte-identity check already exists downstream — this gate fires BEFORE 17 minutes of encode.
3. **Ledger-persisted-before-dump preflight.** A STRICT preflight (`src/tac/preflight.py`, claim a catalog number via `tools/claim_catalog_number.py claim`) that scans `experiments/ddm_*.py` producers for the pattern "write optional dump/npz AFTER the ledger row append" and for `np.concatenate(`/`np.stack(` on a possibly-empty accepted list without an emptiness guard; same-line waiver; fix the live instances (rp1's producer first).
4. **Screening-law declaration at charter lint.** `tools/codex_arm_queue.py::lint_charter_optimal_form`: a charter whose OPTIMAL FORM declares a subset/screen SCOPE reduction must also declare the actuator's confinement (`pair-confined` | `global`), and a `global` actuator's subset screen is flagged as a MECHANISM reduction (verdict-invalid) unless a TOY-BRACKET is declared. Warn-only at landing, STRICT-flip in the same batch if live count is 0.

## PRIOR-LAW PREDICTION (m38)
- Gate 1 would have refused the sj1 no-overlay base (500×) and passed every promoted base of moves 31–37 (all within 1.2×). Gate 2 would have refused `ddm_sj1_successor_price.sh` as pinned today. Gate 3's live count on `experiments/ddm_*.py`: predicted 2–6 instances. Gate 4's live count over `.omx/research/charters/`: predicted ≤ 3 (rw1, fe1, rp1 declare screens).
- **FALSIFIER:** if any promoted move's base fails gate 1's band, the band is wrong — widen from measurement, never disable. If gate 2 refuses a legitimate cross-coder pricing (e.g. pricing a NEW coder against the old), the rationale flag is the door; it must not be silent.

## SCOPE
Apparatus only. Each gate = fix + gate + ≥ 12 tests + a row in `docs/meta_bug_class_catalog.md`. No score work; do not run any pricing or pose job beyond a test fixture.

## HARD CONSTRAINTS
- `upstream/` READ-ONLY; never edit `submissions/`; do NOT modify a live arm's running code path in a way that changes its numbers — gates refuse, they never alter values. sj1 gen 3, rp1 and bnd1 are LIVE: land gates 1–2 as opt-in-by-default-ON with an env escape `TAC_INSTRUMENT_GATES=0` for a process already mid-run, and say so in the memo.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 visible review passes (`tools/review_tracker.py mark-file`); ruff clean; tokens `[no-triality] [p0-ledger-ok]`; NEVER a Co-Authored-By or AI-attribution trailer. If the sandbox refuses git object writes, leave the serializer-fallback bundle and say so. Checkpoint `tools/subagent_checkpoint.py --subagent-id ddm_pm2`.
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29).

## PRIOR NEGATIVE SIGNAL
- "Detector zeroes on the cure; structural > procedural" (m100): a note in a memo is not a gate. "Magnitude dismissal" hooks exist for commits, not for measurements — this is the measurement-side sibling.
- Catalog quota: past #400 a new STRICT gate must retire/replace or carry the file-level waiver — check `check_catalog_quota_under_400` and consolidate: gates 1+2 may be ONE umbrella `check_instrument_binds_to_live_pointer`.

## OPTIMAL FORM
- Reference form: the fire-tool sys.path fix + regression test (`6c74c56fd`) and scg1's seal-contract gate (`89f9656f3`) — fix + test + gate in one landing. SCOPE reductions: none. MECHANISM reductions FORBIDDEN: no warn-only-forever (STRICT-flip atomicity rule); no gate without a test that fails on the original incident's shape.
- **PRIOR-LAW PREDICTION (falsifiable):** as above.

## DELIVERABLE
Code + tests + catalog rows + memo `.omx/research/ddm_pm2_instrument_laws_as_structural_gates_20260909.md` (live counts per gate, the incidents each would have caught). Commit via the serializer. End with the live frontier line.
