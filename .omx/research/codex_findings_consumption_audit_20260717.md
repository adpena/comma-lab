# Codex findings-memo consumption audit — 2026-07-17 (ARM F)

**Operator question (2026-07-17, verbatim):** *"What other findings memos were landed that
nobody consumed? That is a super poisonous bug class and i trusted that you were reading and
following up as they landed like you do for Claude subagents."*

**Answer up front:** in the last-14-day window (47 findings memos, adjudicated exhaustively),
**exactly ONE memo is a true ORPHAN**: `codex_findings_harvest_held_catalog406_20260715_codex.md`
(52.6K — the largest recent memo; the exhaustive Catalog #406/#332 strict-flip backfill worklist).
The 4-arm basis cluster the operator caught (curvelet_throughR_p0, genuine_curvelet_shearlet_
build_measure, no_fourier_basis_sweep, optimal_basis_beyond_fourier) was the only other recent
orphan set and is now CONSUMED-VIA-RECOVERY (Arm D + `no_fourier_basis_DAG_FEED_20260715.md`).
The independently-built warn-only preflight gate (see RESOLUTION) re-derives the same answer
mechanically: live count 1 = harvest_held_catalog406. The bulk pre-July eras are classified by
era-absorption with sampling (honesty boundaries below).

Scale audited: 1,087 `codex_findings_*.md` + 192 `codex_session_summary_*.md`;
220 disposition-ledger rows (115 closed / 60 reviewed_committed / 41 held_entangled / 4 respawned).

---

## Classification rule (explicit — the cited-vs-consumed line)

A memo is **CONSUMED** iff a consumer artifact **records a DECISION about its findings**:
- a companion `*_DAG_FEED_*.md` (or a FEED block in the main DAG) that routes, absorbs, builds
  from, **or rejects the findings with a named reason** (a recorded NO-GO with verdict_scope IS
  consumption — negatives are signal);
- a task# / operator-P0 ledger row / spec § / DSL lever / canonical equation derived from it;
- a commit that implements or remediates the findings (not merely cherry-picks the arm's bytes);
- a disposition-ledger reason that itself records the findings verdict (e.g. warm_start
  "verdict spawn-none", sweep_failures "2 fixed+self-protect / 1 deferred").

A memo is **NOT consumed** by:
- a **custody disposition** alone (`reviewed_committed`/`closed` = the arm's *bytes* were
  handled; says nothing about the *findings*) — the proven orphan mechanism;
- a **bare filename/label mention** (e.g. a "Stores consulted" list in a sister memo);
- a sister `codex_findings_*`/`codex_session_summary_*` citation (producer surfaces, not
  consumer surfaces).

**PARTIAL** = some findings routed, named remainder dangling. **ORPHAN** = zero
decision-recording consumers. **ERA-ABSORBED** = the memo's entire paradigm was terminated by a
recorded pivot decision (family-level consumption; per-memo routing not verifiable).

## Method

Per recent memo: label extracted (`codex_findings_<label>_<stamp>[_codex].md`), then searched
across (i) main DAG `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` FEED blocks +
per-arm `sub015_DAG_*`/`*_DAG_FEED_*` files, (ii) all non-codex `.omx/research/*.md` content,
(iii) `.omx/state/operator_p0_ledger.jsonl`, (iv) git log --grep, (v) `src/tac/witness_dsl`,
(vi) the disposition ledger reasons. Ambiguous hits adjudicated by reading the consumer context
(decision vs bare mention). Scan script preserved in session scratchpad; the durable mechanized
version is the new preflight gate (RESOLUTION).

---

## Tier 1 — last 14 days (2026-07-03 → 07-17): 47 memos, EXHAUSTIVE

### ORPHAN (1)

| Rank | Memo | Estimated signal value | Reactivation route (one line) |
|---|---|---|---|
| 1 | `codex_findings_harvest_held_catalog406_20260715_codex.md` | **HIGH** — the exhaustive, per-factory/per-flag identity map of all **3,884** residuals blocking the Catalog #406/#332 strict flip (899 for v9_cgauge_432 + 995×3 for the ideal/core factories: missing Lever owner / LawRef / compiler-record / provenance-rung / runtime-receipt edges, + 6 named compiler/LawRef disagreements + stale provenance key `schedule`), plus PASS verdicts on harness sources A/B | **ROUTE (recorded here, this is the consumption decision):** the CLAUDE.md 2026-07-15 reconciliation already declares the strict flip OWED; this memo's §"Required backfill before strict flip" IS the actionable worklist for that owed item. Whoever executes the #332/#406 backfill MUST start from this memo — do NOT re-derive the 3,884-tuple map. Standing reminder = the new preflight gate (nags until routed/flipped); no separate task created (would duplicate the CLAUDE.md OWED row). |

No other Tier-1 orphans. (Also verified: the memo has NO disposition row at all — the arm's
landing itself slipped the review gate; the delegation exists in
`.omx/state/codex_delegations.jsonl` but the ledger never dispositioned it. Both gate holes are
what the RESOLUTION two-landing closes.)

### CONSUMED-VIA-RECOVERY (4) — the proven orphan class, since recovered by operator memory

`curvelet_throughR_p0`, `genuine_curvelet_shearlet_build_measure`, `no_fourier_basis(_sweep)`,
`optimal_basis_beyond_fourier` — rc=0 REVIEWED with custody dispositions ("pooled into
drained-fleet consolidation pile" / cherry-picks), findings unrouted until the operator caught
it; now consumed via `no_fourier_basis_DAG_FEED_20260715.md` (ban + curvelet opt-in + owed
`curvelet_through_R_dseg_ab`), `curvelet_throughR_p0_DAG_FEED_20260715.md`
(BUILT_LOCAL_VERIFIED_PREPARED_NOT_FIRED chain), memory `no_fourier_basis_DAG_FEED_20260715.md`
hook, and the live Arm D respawn. These 4 are the empirical anchor for the structural fix.

### CONSUMED (42) — consumer citations

Consumption route legend: FEED = decision-recording `*_DAG_FEED_*.md` companion (spot-checked
for decision quality: `curvelet_throughR_p0` FEED carries a full triality route chain;
`bregman_all_surfaces_504` FEED likewise); DAG = main-DAG FEED block; DISP = disposition reason
recording the findings verdict; P0 = operator-P0 ledger row; EQ = canonical equation.

- arm_derive_solver_provenance — FEED (`arm_derive_solver_provenance_DAG_FEED_20260715.md`) + `P0_campaign_queue_20260715.md`
- basis_d21a_prod — FEED + DAG 18044 (PREPARED_NOT_FIRED decision) + DISP
- bregman_all_surfaces_504 / bregman_v9_all_surfaces — FEEDs + DSL absorption (witness_dsl hits)
- c0_optform_compute_audit — FEED + cherry-pick 8df4723df8 (pose-blind gate implemented)
- c1_deepmath_integration — FEED + DISP (d5901a6404)
- costate_organ_router_stability — FEED (`costate_organ_router_stability_DAG_FEED_20260714.md`)
- defect_network_tube_rate_code — DAG 18956 (#452 DOMINATED verdict recorded) + EQ ("Lossless defect-component delta rate law")
- dsl_hash_enforcement — FEED (`dsl_compile_hash_enforcement_DAG_FEED_20260715.md`) + Catalog #406 gate landed
- exp_linear_reparam_warmstart — own `sub015_DAG_exp_linear_reparam_warmstart_20260714.md`
- frozen_segnet_exact_forward — DAG FEED-task456 + EQ `segnet_exact_forward_cpu_thread_control_v1` + trainer RESPONSE row (DAG 16804)
- governor_measured_growth_fix — own `sub015_DAG_governor_measured_growth_fix_20260714.md`
- jrd_coeff_prefix — DAG master FEED (JRD N/A-with-reason) + EQ `jrd_exact_coefficient_prefix_selection_v1`
- jrd_oss_reconciliation / yopo_oss_reconciliation / sfess_oss_reconciliation /
  master_oss_reconciliation — DAG 16540 single master FEED for #449 recording per-family NO-GO
  verdicts with scopes ("VERDICT: none clears task 449 in the registered scopes") + 4 EQs.
  (Note: sfess/yopo/jrd sub-memos have NO direct consumer; they are consumed THROUGH the master
  memo which the master FEED routes — umbrella consumption, verified decision-recording.)
- lsi_tau_anneal_metric_synthesis — FEED + premise-falsification crosscheck memo + DISP f32bcc96fb (honest negative recorded)
- margin_adaptive_mixed_precision / margin_adaptive_perlayer_followon — FEEDs + DSL absorption (4–6 witness_dsl hits)
- n1_lowdata_learning_theory_oss — FEED + `n1_lowdata_learning_theory_organ_20260714.md` + memory hook (n1 capacity ceiling)
- no_fourier_basis (audit arm, 07-15) — FEED + P0 ledger (5 hits) + MEMORY.md hook
- onpolicy_forward_surrogate (+ _joint_control) — DAG FEED-task455 4-block chain (scaffold-seal
  → terminal-correction → methodology-falsification → corrected-campaign; supersessions recorded) + `onpolicy_surrogate_95kill_20260713.md` + P0 `p0_455_465_95kill_wave`
- optimal_metric_p0_surrogate_followons — FEED + build spec + DISP (pooled)
- optimal_metric_training_loss_curriculum — FEED + DSL hit
- p0_backward_closer_postcommit — findings self-closed in code (trust-root registry landed in
  `curriculum_candidate_pool.py`, verified present; 4 findings each with closure recorded;
  review_counter rounds recorded against commit 26977fb41c)
- perclass_convergence_ab — FEED + git (2 commits)
- pre_se_multi_source_reopen — FEED + DSL hits
- recursive_fractal_optimal_representation_v9 — FEED (12.5K) + design memo 503 + build spec
- repoint_dismissed_intake — FEED + `means_audit_4enum_negatives_open_derivation_fourier_20260715.md`
- rgb_cargocult_scrutiny_optimal_replacement — FEED + `cargocult_sweep_pr95_fourier_rgb_20260715.md` + memory hook (RGB=FINISHER-ONLY)
- ripo_fisher_trust_region — FEED + MEASURED memo + optimal-metric unification derivation
- surrogate_vjp_fidelity_metric — FEED + DSL absorption (3 hits)
- sweep_failures_ledger — DISP 60895ca783 ("2 fixed+self-protect / 1 deferred" — deferral decision recorded) + DAG hit
- throughput_authority_ladder / throughput_frontier_math / throughput_nogo_naive_rescope_audit — FEEDs (the nogo FEED 8.7K records the rescope decisions) + memory `feedback_no_naive_implementations_binary_nogo...`
- v9_cgauge_fake_implementation_audit / v9_cgauge_fake_remediation — FEEDs + CLAUDE.md
  "2026-07-15 RECONCILIATION" paragraph (the #501 audit → warn-only gates row)
- warm_start_reactivation_audit — FEED (9.6K) + DISP 927ef10723 ("verdict spawn-none" decision)
- warmstart_gauge_symmetry_homotopy — FEED + BUILD_SPEC + signal_loss_audit
- warmstart_organ_n1_rl — FEED + build spec + memory hook (n1 organ capacity ceiling banked)
- witness_train_sweep_spec — FEED (`witness_train_sweep_spec_DAG_FEED_20260714.md`)

### PARTIAL (0 confirmed, 1 watch item)

- No Tier-1 memo met the PARTIAL bar (named findings dangling while others routed). Watch item:
  the "pooled into drained-fleet consolidation pile" disposition reason (7 arms) is custody-only
  language; those 7 all independently have FEEDs, which is why they classify CONSUMED — but the
  pooled-reason pattern is exactly how a future orphan slips through. The RESOLUTION's
  `--consumed-by` requirement makes that pattern impossible going forward.

## Tier 2/3 — pre-July eras (honesty boundaries explicit)

- **2026-06-03 → 06-07 tail (15 memos: hinerv/snerv/tilde/aurora/wall-attention/submission-
  byte-shaving):** classification **ERA-ABSORBED**. Spot-checked 2/15
  (`hinerv_snerv_interpreter_burndown`, `tilde_parallax_snerv_hinerv`): 0 direct DAG label hits.
  Family-level consumption decision: the 2026-06-25 operator frontier redefinition (CLAUDE.md
  "THE CURRENT FRONTIER... NON-RGB TASK-SPACE WITNESS CAPSTONE") terminated the HiNeRV/SNeRV
  vehicle line with a recorded rationale; per L18 ancestor rule its numbers do not transfer.
  Residual dangling value: LOW (mechanism lessons only). **SAMPLED, not exhaustive.**
- **2026-05-08 → 06-02 bulk (1,024 memos; substrate-wave/MLX-port/PR-intake eras):**
  classification **ERA-ABSORBED / SEED-BASELINE**. Random sample 8/1,024: 1/8 had any non-codex
  research reference; 0/8 in the main DAG by label. These predate the DAG-FEED routing
  discipline (which begins ~07-12); their consumption model was same-session parent harvest,
  declared honestly by the ledger's 103 `seed-baseline` closures ("pre-gate-install baseline;
  harvested in prior session(s) per commit log"). Their paradigms (HNeRV-family substrates,
  0.196–0.199 cluster) were superseded by recorded pivots. Residual dangling value: LOW.
  **SAMPLED (n=8), not exhaustive — a per-memo audit of 1,024 superseded-era memos is not a
  good use of signal-recovery effort; the gate protects the go-forward window instead.**
- **192 `codex_session_summary_*.md`:** producer-side session logs, not findings memos —
  out of the findings-consumption class by construction (their content summarizes work whose
  routing is adjudicated above per-arm). **NOT individually audited.**

---

## RESOLUTION — the structural two-landing (this bug class, permanently fixed AND self-protected)

**THE BUG:** `tools/codex_landing_review_gate.py` dispositions were pure CUSTODY states —
`reviewed_committed`/`closed` required no consumption pointer, so "reviewed" could (and did,
4+1 times) mean "bytes handled, findings rotting."

1. **Write path (landing A):** `reviewed_committed`/`closed` dispositions now REQUIRE
   `--consumed-by <task#|p0-row|DAG-FEED|spec-section|lever|memo-path|commit-sha|none:<reason>>`;
   bare `none`/`n/a`/`tbd` refused; `respawned`/`held_entangled` exempt; old rows
   backwards-compatible; `consumed_by` recorded in every new ledger row.
   Commits: `9a550c200a` (core enforcement, landed by main supervisor on the operator's
   immediate-fix directive) + `3db70045ef` (ARM F: docstring contract, usage example, and 12
   enforcement tests — refusal/placeholder/escape/exemption/backwards-compat/CLI-parse; suite
   27 passed).
2. **Read path (landing B):** `tac.preflight.check_codex_findings_memos_consumed` — WARN-ONLY,
   wired into `preflight_all()`: flags `codex_findings_*.md` younger than 3 days whose arm label
   appears in NO consumer surface (DAG-FEED-style filename / `consumed_by` ledger receipt /
   P0 ledger / recently-touched research content). In-memo waiver
   `CODEX_FINDINGS_CONSUMPTION_WAIVED:<rationale>` with placeholder rejection. 22 tests.
   Commit: `a034d7c9f2`. **Live validation: the gate independently re-derived this audit's
   answer — 11 fresh memos scanned, 1 violation = harvest_held_catalog406.** Strict-flip
   condition: after the #406 worklist routing above lands and the go-forward window stays at
   live count 0 for a full cycle.
3. This memo (landing C, the audit): the orphan's routing decision is recorded in the ORPHAN
   table above — which also drives the gate's live count to 0 the honest way (a
   decision-recording consumer now exists), not by waiver.

## Round-1 adversarial self-review (attack on my own audit)

- **Am I conflating cited with consumed?** The rule §above draws the line explicitly and it was
  applied: `sfess_oss_reconciliation`'s mention in `replace_round5`'s "Stores consulted" was
  REJECTED as consumption (bare mention); it classifies CONSUMED only through the master #449
  FEED's recorded NO-GO. Conversely `defect_network`'s DAG row was ACCEPTED because it records
  the DOMINATED verdict with scope. The distinction was load-bearing in 6 adjudications.
- **Weakest link:** umbrella consumption (yopo/sfess/jrd through the master memo) assumes the
  master FEED's per-family verdicts faithfully summarize the sub-memos; I verified the verdict
  lines exist per family but did not re-derive each sub-memo's full findings against its FEED
  summary line. If a sub-memo carried a secondary finding outside its family verdict, it could
  dangle undetected. Bounded risk: all four are NO-GO-scoped throughput probes.
- **Gate false-clears:** the preflight gate treats ANY recent research-content label hit as a
  consumer — cited-vs-consumed is NOT machine-adjudicated (stated in its docstring). It detects
  total orphanhood only; the audit-memo discipline owns the finer line. A memo could therefore
  be gate-clean yet PARTIAL. Accepted: warn-only detector, human adjudicator.
- **Sampling risk (Tier 2/3):** n=8 of 1,024 gives a wide interval; if the bulk era hides a
  high-value orphan, this audit missed it. Mitigant: those paradigms are dead by recorded pivot;
  the operator's question is answered exactly for the window where findings still have live
  routes. Boundary stated, not hidden.
- **Label-collision risk:** short/generic labels could false-positive-match consumer content
  (e.g. `no_fourier_basis` matching the ban memo IS the consumer, fine; but a generic label like
  `sweep_failures_ledger` could match unrelated text). Every Tier-1 CONSUMED row cites a
  specific consumer artifact I opened or a disposition reason I read, not just a grep count.

*Pointer delta: NONE — apparatus/audit work; contest-CPU pointer unmoved. No score claim.*
