# c2_surgical_warm launch stack — FRESH-EYES RECURSIVE ADVERSARIAL REVIEW (2026-07-16)

**Reviewer:** independent P0 arm (author/reviewer separation; did not write the stack).
**Scope:** the GO-chain surfaces landed 2026-07-16 — `spec_c2_surgical_20260716.py` ·
launcher edits (registration / spawn guard / dry-start sec-ep fix) ·
`warm_start_schedule_reconstruction_v1` · the dry-start retry loop · the receipt→launch path.
**Protocol:** CLAUDE.md recursive adversarial review §1-9 (incl. assumption-challenge +
measured-runnability axes); clean-pass counter `tac.review_counter`
surface `c2_launch_stack_adversarial_review_20260716`; MAX 5 rounds.
**Authority:** $0, read + local tests only. **Pointer 0.19108 UNMOVED — MEANS.**

## HEADLINE (for the coordinator, time-critical)

1. **NOTHING WILL SELF-FIRE.** The retry loop (pid 36597) is **DEAD** — it logged
   attempt-1 rc=4 at 14:12Z and again (second start) at 14:14Z, then died inside its
   `sleep 600` (the classic BG-bash SIGURG kill; the script was not
   `start_new_session`/durable-daemon detached — the exact L48/L49 memory class).
   No attempt-2 row exists; no process is alive. MEASURED (ps + attempts log + clock).
2. **Even alive, the loop could not have fired the real launch.** `_run_dry_start`
   "NEVER proceeds to the real spawn" (by construction, verified at the b3 gate), and
   the retry script exits on rc=0 without any real-launch command. The "armed self-fire"
   model was wrong **in the SAFE direction** — a green receipt only clears the
   `C2_COMPOSED_BENCH_NOT_MEASURED` blocker; the real launch requires a separate
   explicit governed invocation (CONTAINMENT preserved, `operator_go_required: true`).
3. **One REAL config defect found and FIXED (F3, the l7 final-epoch off-by-one).**
   The typed config hash CHANGED: old `994cd28576556098…` → **new
   `6c863e71bc3ab67114f82849912d44728e958dad18c23fea2859e83993a2937e`**. Any relaunch
   of the retry loop must expect the NEW hash; the old pinned hash can never produce a
   clearing receipt (that fail-closed behavior is correct and intended).

## Round 1 findings

| # | Class | Finding | Verdict scope | Status |
|---|---|---|---|---|
| F1 | CRITICAL (progress, fail-safe direction) | Retry-loop daemon dead — BG-bash SIGURG killed it mid `sleep 600`; started twice, died twice. Violates the daemon-durability rule (needs `setsid`/`start_new_session` or `spawn_durable_daemon`). | INSTANCE (this loop invocation) | SURFACED — relaunch is the coordinator's call (durable form) |
| F2 | MEDIUM (model correction) | "Self-fire on green receipt" is structurally false: dry-start exits before the spawn; no wrapper fires the real launch. Safe direction; coordinator's mental model corrected. | INSTANCE | SURFACED |
| F3 | CRITICAL-config / MEDIUM-impact — **CONFIRMED DEFECT, FIXED** | `--l7-start-epoch 1400 == epochs`: the trainer epoch loop is `range(start_epoch, epochs+1)` **INCLUSIVE**, and `_seg_form_for_epoch` returns `l7_softplus` at `ep >= l7_start` ⇒ **l7 (the measured defect stage) RUNS on the final epoch ep1400**, plus a spurious stage-transition moment-reset at 1400. The trainer's own comment (~L15646) documents this exact off-by-one and prescribes `epochs+1`; the mod32cap config of record itself parks l7 at **1001 with epochs=1000** ("TRUE never"). Root cause: the newer DSL epoch-budget gate refused ALL `-start-epoch > epochs`, contradicting the ordering validator + trainer relax (4bf533cab) and squeezing the spec into `== epochs`. Impact bounded (1 epoch; EMA-best by-verdict protected; final non-best ckpt + last verdict epoch contaminated) but it violates the spec's stated intent and lineage. | FORMULATION (the c2 emitted config + the budget gate + the day-old equation law) | **FIXED + class-fixed** (below) |
| F4 | MEDIUM (runbook gap) | Crash-resume rollback trap: launch.sh pins `--resume-from <mod32cap BEST npz> --warm-start-weights-only`; the trainer has NO auto-preference for the out_dir's own `levelset_resume_state.npz`. A naive relaunch of launch.sh after a mid-run crash **rolls back to ep651** and its first rolling ckpt **overwrites the crashed run's resume sidecar** in the same out_dir. The documented caveat ("re-applies weights-only / fresh moments") understates this: it is a full progress rollback, not just a moment reset. Correct resume = `--resume-from <run out_dir>` (dir form prefers the resume sidecar; PASS-2 of the dry-start proves exactly this path works). | INSTANCE (this config's resume procedure) | QUEUED: resume runbook line owed before the real launch (see §Owed) |
| F5 | LOW | `parse_dry_start_run_metrics` counts ANY `ep`/`epoch`-keyed JSONL row — the pre-loop "armed" `lever_engage` rows carry `epoch: <start_epoch>` (651 on this warm start), inflating `epochs_completed` before any step. The green verdict is protected by the `checkpoint_written` requirement, and the sec/ep offset math subtracts `resume_start_epoch-1`, so no green-path corruption — but the parser semantics are fragile (a partial epoch at SIGTERM is also counted, acknowledged in the docstring). | FORMULATION (parser) | SURFACED (non-blocking) |
| F6 | LOW | The receipt glob in the spec factory is `Path("experiments/results").glob(...)` — cwd-relative. From a non-repo cwd it silently finds no receipts (fail-closed, safe direction). | INSTANCE | SURFACED |
| F7 | LOW | Same-outdir spawn guard matches `str(out_dir)` as a substring of process cmdlines — relative-vs-absolute path form could false-negative. psutil IS present (7.2.2) so the fail-open ImportError path is dormant; per-process errors are caught; an unexpected psutil error would crash pre-spawn (fail-closed). | INSTANCE | SURFACED (non-blocking) |
| F8 | INFO | The admission refusal is honest: ambient used 37.2 GiB + projected 71.5 GiB > adaptive ceiling 103.0 GiB. The registry shows ONE live daemon (memory_blackbox). Admission clears when ambient drops ~6 GiB — a rerun of the (durably detached) retry loop is the right instrument. | — | — |
| F9 | MEDIUM (measurement bias, disclosed) | The dry-start receipt's `sec_per_ep` structurally measures ONLY the pre-engage regime (ep651-653): the phase/satisfice/subpix terms engage at 700, pose at ≤1000, Muon at 726 — none within the bounded bench. The wall-clock projection (~1.1-1.9 d at ~121-180 s/ep) is therefore a LOWER-bound-biased estimate for the post-engage majority of the run. A7's "sec/ep unmeasured" blocker is discharged by the receipt only in the pre-engage sense. Mitigation already in-config: component-wallclock telemetry ON at probe-every 1 + holistic check-ins. | FORMULATION (bench scope) | SURFACED (disclose in the launch brief; non-blocking) |

## The F3 fix set (landed this review; small + tested)

1. **`src/tac/witness_dsl/spec_c2_surgical_20260716.py`** — `--l7-start-epoch` =
   `epochs + 1` (base + required-actuation + constants row + banner/A3 comments).
2. **`src/tac/witness_dsl/curriculum_dsl.py::schedule_epoch_budget_violations`** —
   NARROW exemption: `--l7-start-epoch == epochs + 1` (the canonical deliberate-parking
   form; mod32cap record + fresh_seeded precedent). Any other past-budget value — and
   every other stage flag past epochs — is still refused (dead-stage protection intact).
   This is the CLASS fix: the gate was refusing the correct form repo-wide, which is what
   squeezed the spec into the off-by-one.
3. **`src/tac/canonical_equations/evaluators.py`** — `run_length_exclusion` mode now
   returns `run_epochs + 1` (the TRUE never-runs boundary given the inclusive loop);
   docstring records the amendment + the original (off-by-one-reproducing) behavior.
4. **`warm_start_schedule_reconstruction_20260716.py`** — law text, LaTeX, anchor
   (`l7_never_runs_start: 1401`), producers note amended; re-registered (append-only,
   `agent=claude, subagent_id=c2_adversarial_review_20260716`).
5. **Tests** — equation tests updated + the mod32cap-record form (1000→1001) added;
   budget-guard tests updated: fresh_seeded's `epochs+1` parking is now LEGAL (its own
   docstring anticipated this relax); two NEW pure tests prove the exemption is exactly
   `epochs+1` and `l7=1005/epochs=1000` still refuses.
   **All green: 11 (equation) + 53 (budget guard incl. the 2 rewritten + 2 new) = 64 +
   full c2 recompile through the real launcher derive path (budget gate exercised),
   l7 emits 1401, blocker held (fail-closed until a NEW-hash bench receipt).**

### Sister-substrate note (same off-by-one elsewhere; NOT fixed here — out of launch scope)

`curriculum_dsl.py` line ~3929 (`Stage("l7_softplus", "--l7-start-epoch", epochs)` in the
crucible-family factory, comment "l7 PARKED at epochs (no-op tail)") carries the same
`== epochs` misconception. Those configs are not today's launch stack; flagged for the
sister-substrate sweep per the fix-cascade rule.

## Round 2 (verification of the fix + remaining lenses) — CLEAN

* Recompile via `derive_named_config` (runs the epoch-budget gate live): PASS, l7=1401,
  new hash `6c863e71bc3a…`, blocker correctly held.
* `test_c2_spec_constants_rows_recompute` re-runs the evaluator against every one of the
  7 constants rows (machine-checked DERIVED): PASS at 1401.
* Trainer precedent for `l7 = epochs+1`: the mod32cap run of record COMPLETED 1000 epochs
  with l7 parked at 1001 — empirical proof the trainer handles the form (MEASURED, the
  checkpoint c2 warm-starts exists because of it).
* No other consumer keys off `l7_start == epochs` (stage boundaries at 300/726/1401; the
  1401 boundary is never reached; muon-before-l7 WARN row is cosmetic and pre-existing).
* PASS-2 override semantics verified: levers merge LAST (dict update), so the dry-start's
  internal lever REPLACES `--resume-from` (no duplicate flag); `--warm-start-weights-only`
  remains on PASS 2 — on the full PASS-1 sidecar it discards moments, and
  `resume_start_epoch == ckpt_epoch+1` still holds, so the tightened `dry_start_resume_ok`
  contract is satisfied honestly (it proves weight+epoch restore; moments-restore is not
  the c2 contract).
* Warm-start on the BEST deploy npz: documented NO-OP for moments (deploy npz has no opt
  state) → fresh AdamW + `start_epoch = 651` from `rs["epoch"]+1`. `--anneal-epochs 1000`
  verified as the real trainer contract (pins τ/β/LR cosine denominators; run length
  1400 separate; ep>plant holds end values; the `anneal_epochs_WARN` row at boot is
  expected and correct).
* Receipt integrity: report written once at the end (torn/partial JSON → factory's
  try/except skips → fail-closed); hash covers the FULL typed config (gt_cache,
  num_pairs, epochs, every lever) so no cross-config laundering; no-hash legacy receipts
  never clear; `dry_start_boot_ok` requires a written resume ckpt (zero-epoch green
  impossible); sec/ep divide-by-zero guarded (`e <= 0 → None`).
* pa_flipmass artifact present (`reports/pa_edge_weights.json`, 5×5 W_e) with fail-loud
  uniform downgrade; subpix+micro-batch guard fail-closed (c2 default micro-batch=1);
  two-phase pose gating REAL (`--pose-finish-start-epoch 1000` arms pose-blind w_pose=0
  until σ_min plateau or backstop; `--jacobian-basin-telemetry` default ON).

## Round 3 (final re-read of the fix diffs + gate-chain re-trace) — CLEAN

Diffs re-read end-to-end; no stray edit; ruff F clean; review-gate marks recorded;
no test asserts the old behavior anywhere else (`grep run_length_exclusion`,
`l7-start-epoch=1001` refusal test rewritten). Gate chain re-trace on the NEW config:
flag-validate (91 flags — unchanged count, value-only change) → schedule-provenance
(l7 row still DERIVED via the amended law) → budget gate (exempts 1401) → memory
preflight (unchanged inputs) → admission (ambient-dependent) → dry-start (receipt will
record the NEW hash).

## Assumption-challenge answers (mandatory per round)

* **R1:** "The DSL layer's stage-boundary convention matches the trainer's loop
  arithmetic." FALSE — the trainer's loop is inclusive (`epochs+1`), the DSL believed
  `== epochs` is never-runs, and the budget gate enforced the wrong side. This exact
  mismatch was F3. Class-fixed at the gate + law level.
* **R2:** "The dry-start bench transfers to the full run." Only partially — it proves
  boot/step/ckpt/resume + pre-engage sec/ep; it structurally cannot price the ep700+
  phase stack, ep726+ Muon, or ep1000+ pose (F9). The 1.1-1.9 d projection is a
  lower-bound-biased estimate; telemetry + check-ins are the mitigation.
* **R3:** "The mod32cap EMA-best ep650 is the right warm tensor set." MEASURED basis —
  the residual decomposition was made on this exact checkpoint; the BEST npz IS the EMA
  shadow; weights-only semantics are a no-op on it (fresh moments by construction).
  The basin-lock-in risk stays INFERRED (A6) with c1 as the paired fresh arm — honest.
* **R4 (burn risk):** "A crash mid-run is recoverable." TRUE mechanically (resume
  sidecar + PASS-2-proven dir-resume path) but the DEFAULT relaunch command is a
  rollback trap (F4) — the runbook line is the missing piece, owed before fire.

## Owed / recommendations (for the coordinator)

1. **Relaunch the retry loop DURABLY** (setsid/`spawn_durable_daemon`, not BG bash),
   expecting the NEW hash `6c863e71bc3a…`. Admission clears when ambient RAM drops
   (~6 GiB below the 14:14Z snapshot).
2. **Crash-resume runbook line** (F4) in the run dir or the launch brief:
   `--resume-from <out_dir>` (dir form), NEVER re-run launch.sh after a crash.
3. Disclose F9 (pre-engage-only sec/ep) in the #385-style wall-clock brief.
4. Sister sweep of the `== epochs` parking in the crucible-family factory (non-launch).

## Seal state

Round 1 NOT_CLEAN (8 findings, F3 fixed in-review) → Round 2 CLEAN → Round 3 CLEAN.
**Counter at 2 consecutive clean; NOT SEALED (needs 1 more clean round).** MAX-5 budget
respected; the remaining round belongs to a successor pass after the coordinator
decisions on F1/F4 land (fixes are unreviewed new code — my own F3 fix received its
Round-2/3 re-review here, but a fresh-eyes pass over THIS landing is the honest closure).

**Launch-safety verdict: HOLD is the current physical state (nothing can fire: loop dead,
blocker fail-closed on the new hash). SAFE-TO-PROCEED to a durable retry-loop relaunch +
bench on the FIXED config; real launch remains gated on the green NEW-hash receipt +
operator GO, with the F4 runbook line landed first.**

Pointer 0.19108 UNMOVED — everything here is MEANS/apparatus.
