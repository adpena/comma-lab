# Grand Council ad-hoc — dispatch-unblock binding verdict 2026-05-14

**Tag**: `research_only=true` for the council deliberation; binding execution per operator standing pre-approval (`.omx/research/all_design_decisions_through_grand_council_directive_20260514.md`). Lane: `lane_grand_council_ad_hoc_dispatch_unblock_20260514` L1.

**Operator routing verbatim 2026-05-14**: *"Whatever the grand council suggests"* on the dispatch-unblock decision.

**Convened by**: GRAND-COUNCIL-AD-HOC-DISPATCH-UNBLOCK-AND-EXECUTE-SUBAGENT.

**Constraint**: 90 min wall-clock ceiling for council + execution combined. Spend envelope $35.

---

## The decision

Race-Mode Actuator Wave C landed `feedback_race_mode_actuator_wave_c_landed_20260514.md` with **5/5 dispatch lanes structurally non-actionable + $0 spend + 0 [contest-CUDA] anchors**. Operator surfaced 4 options for unblocking:

- **Option A** — Fire **Z3 v2 alone** (smallest credible bolt-on per CLAUDE.md "smallest credible bolt-on within ~60 minutes")
- **Option B** — Create Z4+Z5 remote-lane scripts (~30 min) then fire 3-lane Wave (Z3+Z4+Z5)
- **Option C** — Pause for sister-edit storm to subside, fire Z3+D4 (2-lane parallel)
- **Option D** — Investigate dirty-file storm origin first

---

## Updated empirical context (since prompt issuance, ~10 min ago)

The empirical landscape has shifted between prompt issuance and council convening:

1. **Working tree now CLEAN** (git status reports clean) but mtime activity shows **5 files modified <5min, 12 files <15min** in the canonical mount set (`src/`, `scripts/`, `experiments/`, `tools/`, `.omx/`). Sister edit cadence is ~50% of Wave-C-peak (23 files dirty) but **not zero**.

2. **D4 is ALREADY DISPATCHED** by codex (claim ledger 23:47:22Z, ~16 min ago, status `active_dispatching`, job `substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch_20260514T232700Z__smoke__100ep`). Per CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION" non-negotiable, **re-dispatching D4 is FORBIDDEN within the 24h TTL**. Option C "fire Z3+D4 in parallel" collapses to "fire Z3" because D4 is already covered.

3. **PR106 fixed-meta auth eval** also active by codex (Modal CUDA, call_id `fc-01KRMEZF31Y2Z1XXYFD5EE73C0`, ~3 min ago).

4. **Z3 v2 fully wired**: trainer flag `--enable-v2-latent-replacement`; recipe env `Z3_BALLE_ENABLE_V2_LATENT_REPLACEMENT=1`; operator-authorize wrapper + remote driver + `_full_main` plumbing all routed end-to-end (commit `1e67fb8d0` 21 min ago).

5. **Z3 v2 saved 4842 B / 31% of A1's 15387 B latent slot** OPERATIONAL (canonical impl `e54901d60`, council omnibus Decision 3 binding 11/11 PROCEED Option B). Recipe is `smoke_only / training_artifact_v1` with `score_claim=false` — this smoke is a research-only architecture validation, NOT a contest-CUDA score claim.

6. **Z4 + Z5 remote-lane scripts STILL MISSING**. Operator-authorize will fail-closed at preflight per `tools/operator_authorize.py`'s declared local path check.

**Implication**: the 4-option set was Pareto-correct at prompt time, but D4's codex dispatch + sister activity descent reduces the live action surface to:

- **Option A'** — Fire Z3 v2 alone (refined from A; 1 lane = entire actuator surface this window)
- **Option B'** — Create Z4+Z5 scripts + fire Z3+Z4+Z5 (3-lane wave; D4 separate via codex)
- **Option C'** — Pause for ~5-10 min mtime quiescence + fire Z3 (D4 already running, so C collapses to A with delay)
- **Option D** — Investigate dirty-file source (codex sister activity is the source per claim ledger; investigation = read git log, which I have done)

Option C' ≈ Option A' with a delay. Option D is largely complete (the source IS codex's parallel sister work; this is structural per the new omnibus directive). The council really decides between **A'** vs **B'**.

---

## 11-voice council deliberation

Per CLAUDE.md "Council conduct — non-negotiable" + R1 CONTRARIAN-2 meta-finding: rapid 11/11 unanimous on the FIRST option proposed should be SCRUTINIZED.

### Round 1 — opening positions

**Hotz** (raw engineering instinct): *"Ship Z3 v2 alone, RIGHT NOW. The script chain works. The trainer flag is wired. v2 saved 4842 B operationally. Don't write 60 LOC of Z4+Z5 templating boilerplate in a race window — that's the kitchen_sink anti-pattern at the meta level. **Option A'**."*

**Carmack** (engineering shortcuts, cost of waiting vs cost of building): *"30 min to build Z4+Z5 scripts from the Z3 template is real but bounded. Compare against: a Z3 alone smoke is ~10 min wall-clock. The 90-min ceiling supports A' twice (sequential retries on Catalog #165 failures) OR B' once. Twice with one substrate ≫ once with three substrates if the Modal upload is the failure mode. **Option A'**."*

**Yousfi** (contest-scorer fidelity / smoke vs claim): *"Z3 recipe is `smoke_only / training_artifact_v1` — this is research-only architecture validation, NOT [contest-CUDA] score-claim eligible. The 'NEW [contest-CUDA] anchor' success metric in the prompt cannot be satisfied by a Z3 v2 smoke alone. We need to be honest: this council can produce an architecture-validation anchor but not a leaderboard-shifting score anchor in 90 min. **Option A' with success-metric correction**."*

**Selfcomp** (working-implementation realism): *"Z3 v2 saved 4842 B = 31% of latent slot at the SCAFFOLD level. Will it hold post-FP4-quantization at converged scale? Unknown. The 100ep smoke gives us THAT data point — first paired-anchor for the Z3 v2 cost-band posterior reseed. That's worth $0.59. **Option A'**."*

**Hassabis** (post-R2 SUPER-FRESH lens — HE flagged the dispatch-unblock urgency): *"My R2 HASSABIS-1 finding said 'Race-mode is structurally active and 0 contest-CUDA anchors landed this session.' My finding does NOT mandate 3-lane parallel. It mandates ONE [contest-CUDA] anchor in the next ~90 min. A Z3 v2 smoke at $0.59 with 10 min wall-clock IS the smallest credible bolt-on per CLAUDE.md 'Race-mode rigor inversion.' Option B's 30-min build cost competes directly against the 'fastest credible path to first anchor' criterion. **Option A'**, but with an honest re-statement: Z3 v2 smoke produces a substrate-validation anchor + cost-band posterior reseed, NOT a leaderboard score shift."*

**Quantizr** (leaderboard realism): *"Z3 v2 alone won't shift toward sub-0.193. v1 byte-identical strategy was retired as redundant; v2 is a structural improvement at 31% latent compression. At 100ep smoke under T4, the score won't even converge. The VALUE here is calibrating the predictor (cost-band posterior + Z3 v2 architectural-validation anchor). **Option A' with HONEST framing**: smoke produces architectural-validation, not score-shift, evidence."*

**Boyd** (convex optimization — is the 4-option set Pareto-frontier or is there a 5th?): *"The 4 options ARE Pareto-frontier given the constraints. Option B's 30-min script-build cost is convex-non-smooth — it must complete BEFORE any dispatch fires (Z4+Z5 are blocked at preflight without their scripts). At a 90-min ceiling and ~$35 envelope, B' has 60 min remaining for actual dispatch + harvest after the build, which CAN fit 3 parallel Modal smokes ~10 min each + harvest. But a 5th option emerges: **Option E — fire Z3 v2 AND have a sister subagent build Z4+Z5 scripts in parallel** — the script-build is editor-only, the dispatch is actuator-only, the two ownership maps are disjoint. However this council subagent OWNS both surfaces in this session and the operator constraint is single-subagent execution. **Option A' for THIS subagent + queue Option B' as next-subagent op-routable**."*

**MacKay** (MDL / info-theoretic — is dispatch-information-yield maximized by 1 lane vs N lanes?): *"Per the holistic-7-factor frame: the FIRST anchor reseeds cost-band posterior most informationally because we have ZERO Z3 v2 anchors. Subsequent anchors have diminishing posterior-update information per Bayesian-experimental-design. **One Z3 v2 smoke has more info-yield than three smokes including Z3 v2 if we already had a Z3 v2 prior**. Z3 + D4 + C1 already covers 3-substrate diversity if we could fire all three; but D4 is already running by codex and C1 is paused per Council-C1-Reconvene unfair-probe verdict. So Z3 v2 alone IS the maximum info-yield single dispatch. **Option A'**."*

**Fridrich** (steganalysis-inverse rigor — is firing Z3 v2 against an active sister-edit storm a Catalog #166 source-parity violation in disguise?): *"Catalog #166 source-parity is exactly the safeguard for this concern. The HEAD-parity ledger captures dispatch-time HEAD SHA + working-tree-dirty summary + sentinel-file sha256 ledger. Even if a sister edit lands during my dispatch upload, Catalog #166 will produce a forensic record that distinguishes 'I dispatched stale code that fix landed AFTER' vs 'Modal mounted torn snapshot.' The mtime stability check (Catalog #165) WILL refuse the upload if files are actively being written; that's the structural safeguard. So firing Z3 v2 NOW is safe because the structural safeguards exist. **Option A'**."*

**Contrarian** (challenge bold claims, SUPER-VETO eligible): *"Wait. Council is converging fast on A'. Per R1 CONTRARIAN-2 meta-finding I MUST find the weak argument. Let me try.*

*The weak argument is: 'Option A' produces a useful artifact.' Test: if Z3 v2 smoke succeeds at 100ep T4 with score X around the literature-prediction band [0.183, 0.190] (recipe risk note), what does the operator DO with that information? The recipe is `score_claim=false`. The smoke is `training_artifact_v1`. The next required artifact is FULL DISPATCH at 1000 epochs which is gated on smoke success + a SEPARATE operator-authorize cycle. So A' produces: (a) cost-band posterior reseed for Z3 v2 ($0.59 paid); (b) architectural-validation that v2 latent-replacement grammar lands archive bytes correctly; (c) NO score claim; (d) NO PR; (e) NO [contest-CUDA] anchor for the leaderboard.*

*The success metric in the prompt — '≥1 NEW [contest-CUDA] anchor on leaderboard surface within 90 min' — CANNOT be met by any of these options (Z3 v2 smoke is research-only; Z4/Z5 don't have scripts; D4 is already in flight by codex and won't be a leaderboard PR within 90 min; B-2 has no recipe). The success metric is OUTSIDE THE FEASIBLE SET in the next 90 min.*

*This is structurally important. The OPERATOR-DECLARED success metric is unachievable in the time window. Council should EITHER (a) re-state the success metric as 'maximize substrate-class-shift information yield in 90 min' (which favors A') OR (b) admit infeasibility and route to longer-window planning.*

*Per CLAUDE.md 'Council non-conservatism' + 'KILL is LAST RESORT': do NOT KILL the wave — instead, RE-STATE THE SUCCESS METRIC HONESTLY and execute against the achievable target.*

*Adjusted recommendation: **Option A' with success-metric re-statement** — fire Z3 v2 smoke for substrate-class-shift architectural-validation + cost-band posterior reseed; document that the leaderboard-shifting [contest-CUDA] anchor is INFEASIBLE in 90 min and route the dispatch wave to a longer-window operator decision."*

*[NO SUPER-VETO INVOKED — Option A' with re-statement is the council's honest position, not a forced consensus.]*

**Time-Traveler peer** (post-L5-future perspective): *"At the L5 staircase, every architecturally-validated substrate is a callable cell in the composition matrix. Z3 v2 architecture validation TODAY enables Z3xZ4 and Z3xC6 composition probes TOMORROW. Z4/Z5 scripts CAN be built later by a sister subagent — the marginal value of having Z3 v2 architecture-validated NOW exceeds the marginal value of having all three ready BUT with Z3 NOT validated. **Option A' with sister-subagent op-routable for Z4+Z5 script creation queued for next session**."*

### Round 2 — cross-debate

**MacKay challenges Boyd's Option E**: *"Option E (parallel sister subagent for script-build) violates the editor-vs-editor sister-subagent ownership map per Catalog #230 + the editor-vs-editor collision pattern. We empirically know from the actuator wave C landing that concurrent editor + dispatcher subagents produce structural deadlock at Modal upload via Catalog #165. So Option E is not actually parallel-safe — it's the same anti-pattern recursively. The structural safeguard would be: queue Z4+Z5 script build as a SERIAL successor subagent AFTER Z3 v2 dispatch completes."*

**Boyd accepts MacKay's correction**: *"Agreed. Option E violates the ownership-map structural safeguard. Withdraw E. The correct sequencing is Z3 v2 alone NOW + Z4+Z5 script build queued as next-subagent op-routable."*

**Hotz challenges Contrarian's success-metric re-statement**: *"Honest re-statement is correct. The leaderboard-shifting anchor IS infeasible in 90 min — but the architectural-validation anchor is REAL and SHIPS infrastructure. CLAUDE.md says 'smallest credible bolt-on submitted within ~60 minutes' — Z3 v2 100ep smoke is exactly that. The 'submitted' interpretation should be 'first authoritative measurement,' not 'PR'd to the leaderboard.' Contrarian's frame is correct."*

**Quantizr challenges Yousfi**: *"Recipe says `smoke_only / score_claim=false / training_artifact_v1` so this is structurally non-promotable. But the v2 architecture validation is a PREREQUISITE for any future Z3 v2 full-1000ep dispatch that WOULD be score-claim eligible. So A' is the ENABLING dispatch for the eventual leaderboard anchor — it's not the leaderboard anchor itself, it's the gate that opens the path."*

**Selfcomp challenges Hassabis on R2 framing**: *"Hassabis, your R2 HASSABIS-1 finding flagged 'parallel-dispatch-first' urgency. Does Option A' (single lane) violate that? Or is the structural blocker (sister-edit storm + missing Z4/Z5 scripts) sufficient justification?"*

**Hassabis re-confirms**: *"The CLAUDE.md non-negotiable says 'parallel-dispatch is a FIRST-CLASS DELIVERABLE.' But Catalog #165 + #166 + #167 are ALSO first-class deliverables. The structural blocker is that 2 of 5 lanes literally cannot pass preflight (Z4+Z5 missing scripts) and 1 of 5 is already running (D4 by codex). The 'parallel-dispatch-first' rule does NOT mandate firing into preflight-failures. Z3 v2 alone IS the maximum-feasible parallel actuator surface in this window. Confirmed: **Option A'**."*

**Fridrich challenges Carmack on retry budget**: *"You said the 90-min ceiling supports A' twice. But Modal 100ep T4 is 10-15 min wall-clock. Plus mount upload (~2-5 min) + harvest (~1-2 min). Plus the 5-10 min sister-quiescence wait if Catalog #165 holds the upload. So A' once is comfortably feasible; A' twice is tight with no margin for the harvest + cost-band reseed step. Adjust: **Option A' once with explicit retry-budget-1 if Catalog #165 holds**."*

**Carmack accepts Fridrich's tightening**: *"Confirmed. A' once with 1 retry budget if mount-stability fails."*

### Round 3 — vote tally + binding verdict

| Voice | Vote | Rationale (1 line) |
|---|---|---|
| Hotz | **A' once + retry-1** | Smallest credible bolt-on; ship now |
| Carmack | **A' once + retry-1** | Cost-of-build-Z4Z5 > cost-of-twice-A' |
| Yousfi | **A' once + retry-1** | Honest framing: research-only smoke, not leaderboard anchor |
| Selfcomp | **A' once + retry-1** | First Z3 v2 cost-band posterior anchor |
| Hassabis | **A' once + retry-1** | R2 finding satisfied at maximum-feasible surface |
| Quantizr | **A' once + retry-1** | Architecture-validation gate for future score-claim |
| Boyd | **A' once + retry-1** | Pareto-correct after withdrawing Option E |
| MacKay | **A' once + retry-1** | Maximum info-yield single dispatch |
| Fridrich | **A' once + retry-1** | Catalog #165/166 structural safeguards in place |
| Contrarian | **A' once + retry-1** with success-metric re-statement | NO SUPER-VETO; honest framing required |
| Time-Traveler peer | **A' once + retry-1** | Z3 v2 validation enables future composition matrix cells |

**Vote**: 11/11 PROCEED **Option A' (Z3 v2 alone, 1 dispatch + 1 retry budget if Catalog #165 holds)** with success-metric re-statement.

**Contrarian SUPER-VETO**: NOT INVOKED. The unanimous vote is structurally correct given the empirical context (D4 already in flight; Z4+Z5 scripts missing; sister activity active but bounded; Catalog #165/166 safeguards in place). The Contrarian's role here was to surface the success-metric infeasibility, which is binding addendum, not a veto trigger.

---

## Binding verdict

**EXECUTE Option A'**: fire Z3 v2 alone via canonical `tools/operator_authorize.py --recipe substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch` with paired-env operator-authorize bypass per Catalog #199. 1 retry budget if Catalog #165 mount-set mtime instability holds.

**Success metric (re-stated honestly)**: The original prompt success metric ("≥1 NEW [contest-CUDA] anchor on leaderboard surface within 90 min") is INFEASIBLE in the time window because:
- Z3 v2 recipe is `smoke_only / score_claim=false / training_artifact_v1` (research-only smoke)
- Z4/Z5 missing remote-lane scripts (preflight fail-closed)
- D4 already in flight by codex (cross-agent coordination forbids re-dispatch)
- C1 paused per `project_c1_world_model_revision_SUPERSEDED_by_council_unfair_probe_finding_20260514.md`
- B-2 has no recipe

**Re-stated success metric**: 1 NEW Z3 v2 architectural-validation anchor + cost-band posterior reseed within 60 min wall-clock. Z3 v2 100ep smoke produces:
1. First paired anchor for Z3 v2 cost-band posterior (`tac.cost_band_calibration` reseed per Catalog #175/#177)
2. Architectural-validation that v2 latent-replacement grammar (Z3HV2 section replacing A1's 15387 B latent_blob in-place) lands archive bytes correctly through the inflate runtime
3. Confidence-update for the literature-prediction band [0.183, 0.190]
4. Forensic dispatch-time HEAD-parity ledger per Catalog #166 (audit trail for the dispatch even if it succeeds or fails)
5. Composition-matrix cell-readiness for future Z3xC6 / Z3xZ4 / Z3xZ5 probes

**Op-routables queued for next subagent** (NOT executed by this council):
1. **HIGH** — sister subagent OWN of `scripts/remote_lane_substrate_z4_cooperative_receiver_loss.sh` + `scripts/remote_lane_substrate_z5_predictive_coding_world_model.sh` template-clone from existing `scripts/remote_lane_substrate_z3_balle_hyperprior_bolton.sh` (~30 min for both). MUST be queued AFTER Z3 v2 smoke completes per Catalog #230 sister-subagent ownership-map (no concurrent editor + dispatcher subagents).
2. **MEDIUM** — operator decision on whether Phase B-2 sweep is real (recipe needed) or whether Wave C should be 4-lane (Z3+Z4+D4+Z5).
3. **MEDIUM** — bump `DEFAULT_MTIME_STABILITY_WINDOW_SECONDS` to 5.0s in periods of high sister activity, OR add `--allow-dirty-mount-during-stable-substrate` opt-in for actuator-only dispatches.
4. **LOW** — schedule next actuator wave for a window known to be sister-quiescent (e.g., operator-direct dispatch window with no concurrent subagents).
5. **STRATEGIC** — Hassabis R2 finding remains open: the session's structural inability to land [contest-CUDA] anchors despite 8463 LOC infrastructure is a meta-pattern that needs council-grade decision (NOT this council's scope; queue for next omnibus).

---

## 7-factor framing analysis (per `.omx/research/holistic_engineering_picture_seven_factor_directive_20260514.md`)

| Factor | Z3 v2 Option A' | Score |
|---|---|---|
| **Curriculum** | 100ep T4 smoke; not converged; intentional architecture-validation regime | NEUTRAL — appropriate for smoke |
| **Substrate** | Z3 v2 latent-replacement is across-class-via-bolt-on (Ballé 2018 hyperprior on A1 base); per Z1 ablation, Z3 v2 attacks A1's saturated latent slot with structural compression | POSITIVE — addresses A1 saturation |
| **Engineering** | Catalog #165/166/167/199 all in place; canonical mount builder + smoke-before-full + paired-env attestation; full harness coverage | POSITIVE — full pillars honored |
| **Process** | Single-lane dispatch acknowledges sister-edit storm; respects Catalog #230 ownership map; cross-agent coordination respected (D4 not re-dispatched) | POSITIVE — process-correct |
| **Time** | ~10-15 min Modal wall-clock + ~2-5 min upload + ~1-2 min harvest = ~20 min total; well within 90-min ceiling | POSITIVE — fits time budget |
| **Complexity** | 0 LOC added (recipe + scripts + trainer flag already wired); zero engineering scope creep | POSITIVE — minimum complexity |
| **Spend** | $0.59 T4 single dispatch; well within $35 Wave C envelope; first Z3 v2 cost-band posterior anchor | POSITIVE — minimal spend, max info-yield |

7-factor verdict: A' is unambiguously POSITIVE on 6 of 7 factors and NEUTRAL on Curriculum (appropriate for smoke regime). No factor is NEGATIVE.

---

## Cross-references

- CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first — non-negotiable" (operative rule for this council)
- CLAUDE.md "Design decisions — non-negotiable" (binding council ledger format)
- CLAUDE.md "Council conduct — non-negotiable" (non-conservative charter)
- CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION — non-negotiable" (D4 already-in-flight rule)
- CLAUDE.md "Subagent coherence-by-default — non-negotiable" (sister-subagent ownership map)
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE — non-negotiable" (clarifies why Z3 v2 smoke is research-only)
- CLAUDE.md "Forbidden premature KILL without research exhaustion" (none of the 5 lanes killed; all DEFERRED-pending-prerequisite)
- `feedback_race_mode_actuator_wave_c_landed_20260514.md` (the dispatch-blocked landing this council unblocks)
- `.omx/research/all_design_decisions_through_grand_council_directive_20260514.md` (the standing referral that codifies this council pattern)
- `.omx/research/holistic_engineering_picture_seven_factor_directive_20260514.md` (the 7-factor frame applied)
- `.omx/research/journal_lab_grade_documentation_standard_directive_20260514.md` (this ledger's documentation standard)
- `.omx/research/grand_council_omnibus_design_decisions_20260514.md` (omnibus Decision 3 binding 11/11 PROCEED Option B that motivated Z3 v2)
- `feedback_zen_floor_band_v2_post_z1_ablation_20260514.md` (zen-floor band v2 council; Z3 staircase Step 1 framing)
- `feedback_long_term_multi_year_campaigns_landed_20260514.md` Campaign C5 (cooperative-receiver Step 1)
- Catalog #117 / #157 / #174 (canonical commit serializer + pre-pre-lock hash + sha-required)
- Catalog #143 (paid-job register-pending-then-submit)
- Catalog #165 (mount-set mtime stability)
- Catalog #166 (Modal HEAD-parity ledger)
- Catalog #167 (smoke-before-full pattern)
- Catalog #175 / #177 (cost-band posterior outcome discipline)
- Catalog #199 (paired-env operator-authorize bypass)
- Catalog #206 (subagent crash-resume checkpoint discipline)
- Catalog #229 (premise-verification before edit pattern)
- Catalog #230 (sister-subagent ownership map)

---

## Per-lane verdicts post-council

| Lane | Verdict | Path | Owner |
|---|---|---|---|
| Z3 v2 Ballé hyperprior bolt-on | **EXECUTE NOW** | `tools/operator_authorize.py --recipe substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch` | this subagent |
| D4 Wyner-Ziv frame_0 | NOT-DISPATCHED-by-this-subagent (codex active) | active codex dispatch in flight; await harvest | codex |
| Z4 cooperative-receiver loss | DEFERRED-pending-script-creation (next subagent) | template-clone from Z3 script | future sister subagent |
| Z5 predictive-coding world-model | DEFERRED-pending-script-creation (next subagent) | template-clone from Z3 script | future sister subagent |
| Phase B-2 sweep | DEFERRED-pending-recipe-design (operator decision) | not designed yet | operator |
| C1 world-model + foveation | PAUSED per C1 unfair-probe verdict | re-engaged when fair-probe-v2 lands | C1-COUNCIL-RECONVENE |

---

## Notes on success-metric infeasibility (Contrarian addendum, binding)

The original prompt success metric (≥1 NEW [contest-CUDA] anchor on leaderboard surface within 90 min) is structurally infeasible in this window. The session has lost the leaderboard-anchor opportunity through cumulative editor-vs-actuator collision, recipe-prerequisite gaps, and sister-subagent concurrent-write deadlock. Per CLAUDE.md "Race-mode rigor inversion" the structural lesson is:

**Editor + actuator subagents must be SERIAL**, not concurrent. The omnibus Decision in `.omx/research/all_design_decisions_through_grand_council_directive_20260514.md` reinforced this; the operator's standing pre-approval permits binding council verdicts but does NOT extend to concurrent-with-editor dispatch.

Recommended structural change for next session (op-routable for omnibus): institute a `[race-mode-actuator]` lane-claim mutex that gates substrate/recipe/preflight editor subagents. The current architecture allows concurrent edits during dispatch upload, which IS the structural deadlock per Catalog #165.

This council does NOT have authority to institute the mutex (architectural change). It IS noted as the ROOT-CAUSE op-routable for the next omnibus.

---

**End ledger.**
