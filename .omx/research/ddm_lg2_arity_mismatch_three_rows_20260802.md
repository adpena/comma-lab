---
schema: ddm_lg2_arity_mismatch.v1
date_utc: 2026-08-02
arm: ddm_lg2 (respawn — the predecessor died on a provider weekly limit; nothing was assumed landed)
lane_id: "lane_ddm_lg2_20260802"
research_only: true
score_claim: false
promotion_eligible: false
axis: "[macOS-CPU advisory — source re-derivation + landed-receipt replay. NO training, NO scorer job, NO paid dispatch, NO pointer mutation]"
verdict_scope: INSTANCE
council_predicted_mission_contribution: apparatus_maintenance
consumes:
  - src/tac/preflight.py (check_lane_scripts_have_e2e_smoke_proof; preflight_all)
  - experiments/canonical_local_auth_eval_smoke.py (the evidence function)
  - .omx/state/lane_e2e_smoke_proofs.json (205 stored proofs — primary data)
  - tools/supervise_ddm_b4s_burn4.py (the rollback-and-raise ladder)
  - src/tac/optimization/lane_guard.py (ddm_lg1 #808 + ddm_bs2 ratchet)
  - /Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_0{1,2,3}/telemetry.jsonl (64 lane_guard rows)
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/pw1/pw1_arms.jsonl (600 pair probe receipts)
  - .omx/research/{ddm_b4s_guard_audit,ddm_bs2_lane_guard_schedule_and_binary_occupancy_sweep,ddm_pw1_pose_menu_saturation,ddm_wi1_wrong_instrument_sweep,ddm_lg1_lane_guard}_2026*.md
consumers: [MAIN, the #821 owner, the #822 owner, any burn-5 guard decision, ddm_pw1 successor]
tokens: [no-triality, p0-ledger-ok]
---

# ddm_lg2 — three rows, one genus: **ARITY MISMATCH**

## §0 POINTER HONESTY FIRST

**The exact frontier did NOT move.** `0.1910828242 [contest-CPU]` unchanged; own-vehicle line
`v4d 0.9639878 → pw1 0.9476091` unchanged by this unit. Zero training, zero scorer jobs, zero paid
dispatch. Every number below is `[macOS-CPU advisory]`, `score_claim=false`. This is apparatus —
MEANS, not END. Relative significance where quoted: gap-to-bar `0.7754681` from `S=0.9476091`
against `0.172141`; **1% of gap = 0.0077547 S**.

## §1 STATE CORRECTIONS — my charter was stale on two of three rows

I am a respawn and was told to assume nothing landed. The opposite was true: **more had landed than
my charter knew.** Recorded before anything else, because re-deriving settled work is the cardinal sin.

| charter claim | status re-derived at source |
|---|---|
| #822 = "the λ_Lane budget never tightens; derive whether it *should*" | **ALREADY BUILT.** `ddm_bs2` (commit landed 2026-08-01) shipped the monotone ratchet — `derive_ratchet_budget`, `derive_noise_floor`, `derive_deadband_k`, `calibrate_deadband_k`, `LaneGuardConfig.budget_ratchet`, plus an `inert_slack_gates`/`inertness_alarm` self-report. I did **not** rebuild it. Row B's *second* leg — the escalation ladder — is what remained, and is §3. |
| #822's registered title | Neither my charter's reading nor bs2's is the ledger's. The ledger title is *"lane-guard sign disagreement: `realized_lane_s_units` vs `net_betti0_realized_lane_delta` … TEST whether the budget metric is self-defeating"*. bs2 tested and **REFUTED** it at r = +0.9697 (INSTANCE-scoped). Three readings of one id; the id is over-subscribed, not the work. |
| #871 = the binary-operations sweep | **bs2 already swept it** (84 discrete choice points over 10 live-chain files) under the same id. My row C is therefore an EXTENSION of bs2's denominator, not a re-run — §4. |
| #821 = "the Lane smoke-proof gate is STRICT, 184-violating, unreachable" | **All three legs confirmed by measurement** (§2). `ddm_wi1` had already typed the gate as a wrong-instrument specimen (its cited line numbers `:28416`/`:5653` have since drifted to `:28602`/`:5753`). The **scope predicate** my charter asked for is new here. |

## §2 ROW A (#821) — the gate's evidence function does not read its scope variable

### 2.1 The three legs, MEASURED

| leg | measurement | value |
|---|---|---|
| strictness | `src/tac/preflight.py:5753` | `strict=True`; **POSITIVE CONTROL**: calling it strict raises `MetaBugViolation` (a gate never shown to fire is untrusted) |
| violations | `check_lane_scripts_have_e2e_smoke_proof(strict=False)` | **184 of 204** scanned; `proven=0`, `waived=20` |
| violation *kind* | classified all 184 | **184/184 `stale_proof`**; `0` missing, `0` malformed, `0` corrupt |
| reachability | enclosing block | inside `if check_codebase:` (`:1957`). The commit hook defaults `--no-codebase` (`tools/preflight_hook.py:637`) ⇒ **never runs on a commit**. It DOES run under `PREFLIGHT_FULL=1` and at `scripts/deploy_vastai.py:134` (`check_codebase=True`) — so on those two paths the repo is currently REFUSED. |

The first surprise is in the *kind* column. Every lane script already has a proof; not one is missing.
**The gate's live signal is 100% age and 0% absence** — it is a staleness timer, not a smoke gate.

### 2.2 The decisive fact: the evidence is lane-independent

All 205 stored proofs carry **one** `archive_sha256`, **one** `fixture_archive`
(`experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip`), **one** `stages_passed` tuple, and a
combined `elapsed_seconds` of **10.208 s for 205 "E2E smokes"**, written in an 11-second window
(`2026-07-14T23:22:51Z → 23:23:02Z`, 12 distinct stamps).

Verified at source, not inferred: in `experiments/canonical_local_auth_eval_smoke.py`,
`smoke_archive(archive, lane_name, …)` uses `lane_name` at **exactly one line** — `:486`,
`"lane_name": lane_name`, a label in the emitted dict. **Zero stage functions take it.** Stages 1–4
depend on the archive; stages 5–10 are labelled in-source (`:472`) *"Static stages — independent of
fixture, scan the local repo."*

So the evidence is `f(archive, repo_state)`. The scope variable is not an input.

> **184 violations is not 184 problems. It is ONE fact, fanned out 184 times.**

### 2.3 The derived scope predicate (and the trap it avoids)

A gate's honest population is the number of **distinct evidence tuples**, not the number of scope
items it iterates. Measured here: `fan_out = 205 / 1 = 205×`.

This settles strict-vs-warn **without tuning**, and it disqualifies the obvious fix:

- **Re-stamping is the trap.** Re-running the backfill costs ~10 s and yields 184 fresh green rows
  that assert nothing new. It converts a loud-but-wrong instrument into a **silent**-and-wrong one —
  strictly worse, and it is the same vacuity genus (`skip == green`) the repo cured elsewhere on
  2026-08-01.
- **Recency is contaminated too.** The naive alternative predicate — "scope to recently-touched
  lanes" — reads 194/204 scripts as ≤30 days old, but git shows that is one bulk `.sh` portability
  sweep on 2026-07-07 in 14 batches of 20 (`0e0a128ea7` … `31fafe0791`). Both candidate predicates
  key on a timestamp a bulk maintenance pass can reset. **A predicate a bulk pass can fake is a
  bulk-event detector, not a scope predicate.**

**The predicate that falls out, in two terms:**

1. **GLOBAL leg — population 1, legitimately STRICT.** "The canonical smoke fixture still passes its
   10 stages, ≤ 7 d." One real fact, re-established by one 0.04 s command. Strictness on a population
   of 1 is honest.
2. **PER-LANE leg — report `k of N`, never pass/fail.** Meaningful only where the proof is
   lane-specific, i.e. carries THAT lane's `archive_sha256`. The tool already supports it
   (`--fixture-archive`, `:528`) and 454 `experiments/results/*/archive*.zip` exist locally, so it is
   buildable. **Measured coverage today: 0 of 205.** And 0 is `VACUOUS`, not clean — a gate scoped to
   "lanes with a lane-specific proof" would examine 0 items and print PASSED, which is the same bug
   on the other side. The honest emission is the fraction, not a verdict.

**Recommendation (not executed — see §6):** demote the per-lane iteration to a reported coverage
fraction; keep STRICT only on the population-1 global leg. Do **not** re-stamp.

## §3 ROW B (#822) — the escalation ladder's rung equals its own erasure

The budget schedule was bs2's. What nobody paid is the **ladder**: `tools/supervise_ddm_b4s_burn4.py`
`:791-794`, on a LANE_EROSION rollback, sets
`new_init = min(last_λ + LG1_LAMBDA_STEP, λ_max)` and relaunches with `--lane-guard-lambda-init`,
escalating to an operator ALARM at `LG1_LAMBDA_ESCALATE = 1.0`.

The dual is a clipped rectified integrator:
`step = clip(η·g, ±cap)`, `λ ← clip(λ + step, 0, λ_max)`.

MEASURED over the 64 landed `lane_guard` gate rows (burn-4 windows 01–03, ep644→945):

| quantity | measured |
|---|---|
| `LG1_LAMBDA_STEP` (rung) vs `lambda_step_cap` | 0.1 vs 0.1 ⇒ **rung/cap = 1.000** |
| gates where `|η·g| ≥ cap` (one gate erases a full rung) | **64 / 64 = 100%**; \|η·g\| min 0.2286, median 1.6808, max 3.554 |
| gates with `g > 0` | **0 / 64** |
| replay of the shipped `dual_ascent` from `λ_init = 0.1` on the real `g` series | `λ = 0.0` **after gate 1**; `λ > 0` on **0 / 64** gates |
| gates of sustained violation needed to reach `escalate_at = 1.0` | `1.0 / 0.1 = 10` consecutive `g > 0` gates |

**Derived verdict: `LADDER_INERT`.** `--lane-guard-lambda-init` is an **initial condition on a
contracting map, not a floor.** Because `_last_lane_guard_lambda` reads the last λ of the rolled-back
window — always 0 — `new_init` is always `0.0 + 0.1 = 0.1`: rung 1, forever. The operator-escalation
rung at λ ≥ 1.0 needs 9 consecutive `g > 0` gates; measured `g > 0` on 0/64. **The escalation rung is
unreachable by construction under a constant budget.**

This **subsumes and sharpens** `ddm_b4s_guard_audit` §6 item 2 ("gate the rollback-and-raise on the
dual's own `g`; raising λ when `g < 0` is not KKT-licensed — one-line precondition"). Correct, but
the measurement shows the raise is not merely un-licensed: **it is a measured no-op.** Adding the
precondition changes nothing observable, because the thing it prevents already has zero effect.
`burn4.ALARM`'s `raise_lambda_init_to: 0.1` reads as *action taken* and is not.

**The KKT-correct repair is upstream, and it is already built.** The ladder cannot be fixed by tuning
the rung (raising it above one cap would be a λ *floor*, which is not KKT-licensed while `g < 0`).
The only licensed way to give the dual authority is to let `g` become positive — which is exactly
bs2's `budget_ratchet`. Under the ratchet the dual climbs on its own and the supervisor's raise is
redundant. **Under either regime the ladder contributes nothing, so it should be RETIRED rather than
tuned** — an escalation channel that cannot escalate is worse than none, because its ALARM record
reads as an action. (Governance-knob framing:
`governance_knobs_are_unladdered_control_provenance_20260801`.)

`verdict_scope`: **INSTANCE** for the 64-gate series; **DERIVED** (arithmetic, not sampling) for
"rung = cap ⇒ erased in one gate whenever `|η·g| ≥ cap`" — that leg holds for any series with those
constants.

## §4 ROW C (#871) — the binary sweep, extended by one measurement bs2 called unmeasurable

bs2 inventoried **84** discrete choice points over 10 live-chain files and measured 5. I did not
re-run that. Its three explicit UNMEASURED rows were `token_ste ∈ {round, dither}`, the GN
line-search `scale ∈ (1.0, 0.5)` (5 sites, no telemetry), and the **first-improving bracket
direction**, which bs2 classified *"UNMEASURED — and unmeasurable without adding telemetry."*

**It was measurable.** The telemetry already exists: `pw1_arms.jsonl` records every probe with its
`phase`. Zero new compute.

Confirmed at source (`experiments/ddm_v4d_resolve.py:215-220`, sister at `:334-336`):

```python
step, direction = DIM0_STEP0, 0.0
for sign in (1.0, -1.0):
    d, xq = eval_dim0(best_x + sign * step)
    if d < best_d:
        best_d, best_x, direction = d, xq, sign
        break            # <- -1.0 is never evaluated when +1.0 improves at all
```

MEASURED, n = 600 pairs, from the shipped receipt:

| arm | committed to `+` **untested** | of the `d_pose` mass | both directions evaluated | of those, only `−` improved | of those, neither improved |
|---|---:|---:|---:|---:|---:|
| A (dim0) | **94 (15.67%)** | 12.28% | 506 | 53 | 453 |
| B (beta) | **31 (5.17%)** | 8.21% | 569 | **60** | 509 |
| union | **109 (18.2%)** | **15.89%** | — | — | — |

**Threshold self-check (R1 self-review).** The "which direction improved" columns use `d_ctrl` as a
proxy for the bracket's entry `best_d`, which the receipt does not store. The `break` makes that proxy
*testable*: a first probe that truly improved leaves exactly ONE probe, so any first-probe
"improvement" appearing in the both-evaluated bucket would prove the threshold wrong. **MEASURED: 0 on
both arms** (`threshold_proxy_valid: true`), so the two columns stand. The untested-commitment counts
(94 / 31 / 109) depend on no threshold at all and are exact regardless.

The asymmetry is the finding: **in arm B the `−` direction wins nearly 2× as often as `+` among the
pairs where `−` was allowed to compete (60 vs 31)** — yet the search lets `−` compete only after `+`
has already failed. And pw1's own §3 decomposition MEASURED that the dominant beta win required
`g < −1.0`, i.e. the negative side. **The probe ORDER is biased against the direction pw1 itself
measured as dominant.**

This is the operator's class exactly, one level in from pw1's: pw1 removed a *bound*; the bracket it
installed carries an *asymmetric direction preference*. A two-element ordered iteration with `break`
is a binary whose two points are not symmetrically sampled — the flag's second setting is not a
sample of the continuum, it is a fallback.

**Honest bound on the stake.** The 109 short-circuited pairs have mean `d_pose` **0.006685** vs
population **0.007645** — slightly *below* average. This is a broad, mild under-search, **not** the
tail (pw1 §7: 10 pairs carry 62.1% of the mass). Do not oversell it.

**PRE-REGISTERED NEXT MEASUREMENT (a NAMED REQUEST — I hold no scorer slot and did not fire one):**
drop the `break`, evaluate both initial probes for all 600 pairs, keep the better. Monotone-safe by
construction (same accept rule), so it cannot worsen. Cost: **+94 forwards (arm A) and +31 (arm B)**,
the pairs that currently short-circuit. **Falsifier:** if the symmetric search yields `Δd_pose ≥ −1e-6`
(i.e. no measurable improvement) over the 109 pairs, the probe-order asymmetry is priced at zero on
this vehicle and the row closes at FORMULATION scope.

## §5 WHAT LANDED

- `tools/ddm_lg2_arity_census.py` — three subcommands, one per row, each **reporting its
  denominator** and emitting `VACUOUS` (never `PASS`) on an empty scope. **The instrument is
  persisted, not just the number**: every table above is reproducible by running it.
  - `smoke-scope` → `ARITY_MISMATCH`, fan_out 205×, honest population 1 of 204
  - `ladder-authority --telemetry …` → `LADDER_INERT`, 64/64 saturating, λ→0 at gate 1
  - `bracket-direction --arms-jsonl …` → `UNTESTED_BINARY_COMMITMENT`, 109/600, 15.89% of mass
- `src/tac/tests/test_ddm_lg2_arity_census.py` — **15 tests**, each measurement carrying a
  **POSITIVE CONTROL** that flips its verdict (a ladder that *does* climb; per-lane evidence that is
  *not* flagged; a symmetric search that clears), plus explicit empty-scope `VACUOUS` assertions.

## §6 WHAT I DID NOT DO, AND WHY

- **I did not patch the gate, the ladder, or the bracket.** The burn is over, so a supervisor fix
  changes nothing retroactively (the same reasoning `ddm_b4s_guard_audit` §5 gave); flipping the
  smoke gate's reachability alone would refuse every commit at 184 violations; and patching three
  surfaces I audited inside the same unit is the built-instead-of-paid trap. **The debt is measured,
  named, and routed — not built.**
- **I did not re-stamp `lane_e2e_smoke_proofs.json`.** §2.3: that would make the instrument silent.
- **I ran no scorer job and no training**, per charter. The §4 next measurement is a named request.
- **I did not re-run bs2's 84-row sweep**, and `token_ste`/GN-line-search remain UNMEASURED, not
  cleared.

## §7 verdict_scope ledger

| claim | scope |
|---|---|
| 184/204 violations, all `stale_proof`; 205 proofs, 1 evidence tuple, 10.208 s total | **MEASURED**, this repo state |
| `smoke_archive` does not read `lane_name` in any stage | **VERIFIED AT SOURCE** (`:486` sole use) |
| gate unreachable from the commit hook; reachable via `PREFLIGHT_FULL=1` and `deploy_vastai.py:134` | **MEASURED** (enclosing block + hook default) |
| ladder rung = cap ⇒ erased in one gate when \|η·g\| ≥ cap | **DERIVED** (arithmetic; holds for any such series) |
| 64/64 gates saturate; λ→0 at gate 1; escalation unreachable | **INSTANCE** (burn-4, 64 gates, one run) |
| 109/600 untested binary commitments, 15.89% of mass | **MEASURED** on the shipped v4d/pw1 receipt |
| "the `−` direction would have won more often" | **NOT CLAIMED** — occupancy only; the §4 falsifier is the test |

No prior negative re-opened. No score claim. No pointer mutation.

## STORES CONSULTED

CLAUDE.md · AGENTS.md · `docs/operating_manual_craft_handoff.md` · MEMORY.md (top rows) ·
`.omx/state/main_hot_state.md` · `.omx/state/canonical_task_status.jsonl` (#821/#822; #871 absent
until bs2 registered it) · `.omx/research/ddm_lg1_lane_guard_20260731.md` ·
`ddm_b4s_guard_audit_20260801.md` · `ddm_bs2_lane_guard_schedule_and_binary_occupancy_sweep_20260801.md` ·
`ddm_pw1_pose_menu_saturation_20260801.md` · `ddm_wi1_wrong_instrument_sweep_20260731.md` ·
`ddm_vc1_vacuity_denominator_cure_and_census_20260801.md` · `ddm_gc16_dev_gate_denominator_and_control_ratchet_20260801.md` ·
`docs/meta_bug_class_catalog.md` (#184) · primary code: `src/tac/preflight.py`,
`experiments/canonical_local_auth_eval_smoke.py`, `tools/preflight_hook.py`,
`scripts/deploy_vastai.py`, `tools/supervise_ddm_b4s_burn4.py`,
`src/tac/optimization/lane_guard.py`, `experiments/ddm_v4d_resolve.py` ·
SSD custody `/Volumes/VertigoDataTier/pact/{ddm_b4s_20260731,ddm_v4d_20260731}` ·
memories: vacuity_is_indistinguishable_from_pass · governance_knobs_are_unladdered_control_provenance ·
boolean_flags_are_a_ui_over_a_continuum · built_new_machinery_instead_of_paying_identified_debt ·
negative_existence_claims_are_the_days_dominant_error_class · constants_are_poison · verdict_scope ladder.

**Pointer `0.1910828242 [contest-CPU]` UNMOVED; own-vehicle `0.9476091` UNMOVED.**
`[no-triality] [p0-ledger-ok]`
