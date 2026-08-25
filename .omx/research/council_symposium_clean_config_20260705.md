---
council_tier: T3
council_topic: clean-run config review (first non-confounded n600 witness measurement post-18-confound-cleanup)
council_attendees: [Shannon, Dykstra, Ballé, Assumption-Adversary, Contrarian, PR95-author, Yousfi, Fridrich, Rudin, Daubechies]
council_quorum_met: true
# Catalog #300 v2-frontmatter backfill 2026-08-25: `council_verdict` was authored as
# PENDING_SYNTHESIS while the three benches were still out. This memo's own SYNTHESIS section
# ("═══ SYNTHESIS — T3 VERDICT: PROCEED_WITH_REVISIONS (unanimous 3/3 benches) ═══") records the
# COMPLETED verdict, so the frontmatter was stale relative to its own body. Corrected to the
# body's verdict; the original authored value `PENDING_SYNTHESIS` is preserved verbatim in this
# comment per Catalog #110/#113 HISTORICAL_PROVENANCE. Frontmatter-only; NO body mutation.
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
# council_dissent + council_assumption_adversary_verdict + council_decisions_recorded below are
# transcribed VERBATIM from this memo's own BENCH 1 / BENCH 2 / SYNTHESIS / RECURSIVE
# SELF-REFLECTION sections. Frontmatter-only addition (NO body mutation).
council_dissent:
  - member: Ballé
    verbatim: "eik-OFF baseline risks reading 'won't descend' when the true cause is elsewhere — run eik-off baseline AND eik-0.01 same-seed A/B; the R-roundtrip d_seg gap arbitrates, no load-bearing eikonal verdict from one arm."
  - member: Contrarian
    verbatim: "FORK on the fixed-β=4 edit — is the argv check actually available? [RESOLVED in-memo: witness_autoconfig.py:549 shows annealed-hosc is a SEALED-205 delta, NOT part of proven_base; fixed-β=4 was never the descending config; the edit stands.]"
council_assumption_adversary_verdict:
  - assumption: "`proven_base` is a clean measurement surface after the 18-confound cleanup."
    classification: CARGO-CULTED
    rationale: "BENCH 2 meta-finding, all 3 benches converge: the cleanup fixed the CODE, but two confounds rode the explicit CONFIG FLAGS into proven_base — where the L2 gates (which check argparse DEFAULTS) cannot see them. proven_base is confound-cleaned on 16 axes and re-poisoned on 2. 'Clean of the session's 18 confounds' != 'clean measurement'."
    empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
  - assumption: "--verdict-pairs 96 is an acceptable cost control."
    classification: CARGO-CULTED
    rationale: "CHARGE 1 CONFIRMED VERIFIED-VIA-SOURCE (witness_autoconfig.py:336, :441). Gate #401 passed only because it checks the DEFAULT (0) while the config OVERRIDES to 96. Scoring best-ckpt on 96/600 is 2.5x noisier and induces the optimizer's curse."
    empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
  - assumption: "fixed hosc-β=4 is the config that produced the descending d_seg numbers."
    classification: CARGO-CULTED
    rationale: "CHARGE 2 CONFIRMED VERIFIED-VIA-SOURCE, highest-leverage: levelset `_hosc_beta_for_epoch` returns None without --hosc-beta-end (1982-1985), the loop only updates if non-None (5895) => constant β=4; witness_autoconfig.py:549 shows annealed-hosc is a SEALED-205 DELTA not part of proven_base. The runs that descended used the ANNEALED delta."
    empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
  - assumption: "eikonal-OFF is the optimal baseline."
    classification: INFERRED_FROM_DOMAIN_LITERATURE
    rationale: "Score-orthogonality is sound but the eik-off vs eik-0.01 arbitration is the pre-registered A/B, not yet measured => verdict edit #4 is PROVISIONAL-pending the A/B, correctly framed as baseline+arm, not a kill."
    empirical_verification_status: INFERRED_FROM_DOMAIN_LITERATURE
council_decisions_recorded:
  - "MANDATORY edit 1 (all benches): --verdict-pairs 96 -> --verdict-pairs 0 (n600; kills 2.5x noise + optimizer's-curse selection bias). VERIFIED-VIA-SOURCE."
  - "MANDATORY edit 2: --hosc-beta 4.0 (fixed) -> --hosc-beta 1.0 --hosc-beta-end 4.0 (annealed) OR --activation step_basis. VERIFIED-VIA-SOURCE that the levelset loop runs constant β=4 (CLAUDE.md-forbidden)."
  - "MANDATORY edit 3: --l7-start-epoch 600 -> 1001 (disable the measured d_seg-decoupling defect)."
  - "HIGH edit 4 (bench 1+3): --eikonal-weight 0.01 -> 0 for the BASELINE; Ballé DISSENT recorded — run the eik-off vs eik-0.01 same-seed A/B, the R-roundtrip d_seg gap arbitrates."
  - "HIGH edit 5: --n-dir-freqs 2 -> 4 and --freq-along 4 -> 8 (the #1 measured lever; 3.2x along-tangent deficit; ~0 byte)."
  - "HIGH edit 6: drop --lane-prior-phi1 --lane-prior-phi1-mode replace (measured no-op); keep --structured-init-include-lane and add a part_frac[lane]>0 assert."
  - "KEEP (unanimous): FRESH-from-scratch NOT resume; rollback default; C3 eik-norm; per-group-clip; closed-loop liveness; chroma; w_pose=0 (label d_seg-COMPONENT, no pointer claim); mod-dim 26; softmax-temp anneal; muon@726; length-weight 0.001."
  - "FOLLOW-UP BUG-CLASS FIX (PR95-author): new preflight gate refusing levelset `--activation hosc` without `--hosc-beta-end` or a `# FIXED_BETA_OK:` waiver — extinct the base-vs-levelset safety-default-divergence class permanently."
---

# T3 Grand Council Symposium — clean-run config review (2026-07-05)

**Charge:** review the `proven_base` config + the main agent's open questions before the FIRST non-confounded
n600 witness measurement. Pre-review red flags: `--verdict-pairs 96` (non-n600 subset, C12 at config layer),
`--hosc-beta 4.0` (fixed-β saturation caveat), `--lane-prior-phi1-mode replace` (measured no-op).
Pointer 0.19110 UNMOVED (means).

## BENCH 1 — Shannon-LEAD / Dykstra-CO-LEAD / Ballé (objective/feasibility/RD) — RETURNED
**VERDICT: PROCEED_WITH_REVISIONS.** Apparatus fixes are correct enablers; config re-introduces 2 config-layer
confounds + the resume framing hits TIER-2.

**EDITS (exact):**
1. `--verdict-pairs 96 → 0` (all 600). **Shannon:** scoring best-ckpt + all d_seg telemetry on 96/600 = C12 violation:
   (a) 2.5× noisier (SE√(600/96)) → false GO/ABORT on the gate; (b) OPTIMIZER'S CURSE — argmin over noisy 96-estimates
   selects checkpoints lucky-on-96 whose TRUE n600 d_seg is worse (bias ≈ SE·√(2ln k)). Control cost via CADENCE
   (verdict-every), NEVER pair-subset (chunked n600 is bit-identical + memory-safe). HARD-EARNED.
2. `--eikonal-weight 0.01 → 0` for the BASELINE. **Ballé:** scorer reads the argmax partition = φ's ZERO level set only;
   `|∇φ|=1` everywhere is scorer-INVISIBLE off-boundary + spends the per-group-clip budget on geometry the score never
   reads + JUST caused the poison (C3 unit bug). O(1)-normalized but UNMEASURED at 0.01. eik-0.01 = separate A/B arm,
   NOT a baseline ingredient. "eikonal is a needed SDF stabilizer" = CARGO-CULTED (never measured on THIS scorer).
3. KEEP `--length-weight 0.001` — curve-shortening acts ON the zero level set (the boundary the scorer DOES read),
   suppresses high-freq wiggles that don't survive R; provably non-dominating (0.0002%). Objective-aligned insurance.
4. **FRESH-from-scratch, NOT resume (Dykstra, FATAL).** C10: this config's OWN init levers (structured-init/
   lane-prior-phi1/palette-anchor/chroma) are applied at setup then OVERWRITTEN by the resume model.update → a resumed
   run is a FAKE measurement (levers active:true but inert). Plus C8/C9 stale ep100 moments = preconditioner fit to the
   WRONG (eik-dominated, unit-bugged) Hessian = non-feasible warm-start. Plus the 0.025 "gold" = frozen artifact
   (H2-F4), so "descend past 0.025" tests against corrupted data. Fresh = the only clean convex-feasibility read.
   "faster"=wall-clock; "decisive"=information — optimize decisive (means/ends). De-risked: per-stage ckpts + liveness
   → the gate fires by ~ep100, not 1000.
5. Launcher assert no dup long-flags (C13) before dispatch.

**w_pose=0:** CORRECT for d_seg-isolation (pose rides the stored-target sidecar, blind-by-design in training). MUST
label the run a **d_seg-COMPONENT result** — no full-S/pointer claim; pointer moves only via byte-closed exact eval.

**DISSENT (Ballé, minority, recorded):** eik-OFF baseline risks reading "won't descend" when the true cause is
R-survival jaggedness not a stalled optimizer → run BOTH arms (eik-off baseline vs eik-0.01) same-seed, let the
R-roundtrip d_seg gap arbitrate. No load-bearing eikonal verdict from a single arm (sister of the viscosity-NO-GO
non-load-bearing conclusion). RESOLUTION: baseline=eik-off, queue eik-0.01 as immediate A/B.

**PRE-REGISTERED GATE (all d_seg n600, EMA-through-R, live-stepping epochs; relative to ep0 init):**
- APPARATUS precondition (every epoch or ABORT): accepted-frac ≥0.9 (skip>0.1 for 3ep→ABORT guard-still-deadlocking) ·
  ep_loss>0 (==0→hard ABORT) · eik<40% if on · seg-gnorm not over-clipped >100× (C4 held).
- ep0 record init (~0.02-0.05) · ep50: ≤0.9×ep0 neg-slope else CONCERN · **ep100 PRIMARY: GO if d_seg≤0.008 & <ep50;
  CONCERN 0.008-0.015 neg-slope; ABORT if >0.015 or non-neg-slope despite liveness (live-but-stalled = real vehicle
  problem → attribution, don't burn 900ep)** · ep300 (tau) ≤0.005 · PROMOTE: first per-stage ckpt with n600 d_seg≤0.004
  → byte-close + pose sidecar + upstream/evaluate.py CPU. ONLY exact eval moves the pointer.

## BENCH 3 — Yousfi / Fridrich / Rudin / Daubechies (scorer/representation/curriculum) — RETURNED
**VERDICT: PROCEED_WITH_REVISIONS.** Three MEASURED-axis defects + one measurement-validity risk. Code-grounded.
- **CRITICAL — hosc β runs CONSTANT 4.0.** The base auto-anneals (base:1472/1482) but the LEVELSET loop calls its OWN `_hosc_beta_for_epoch` (levelset:1982-1985) which returns None (no anneal) when `--hosc-beta-end` unset; model built β=4 (2436). Config omits the end ⟹ β pinned 4.0 all 1000 ep = tanh(4·sin)→±0.999 saturation = the CLAUDE.md-FORBIDDEN fixed-β=4 (MORE saturating than the poisoned v5/v6 which DID anneal 1→5.134). FIX: `--activation step_basis` OR add `--hosc-beta 1.0 --hosc-beta-end 4.0`.
- **CRITICAL — l7 RE-ENABLED** (`--l7-start-epoch 600` < 1000 fires it). l7 = MEASURED DEFECT (L∞-in-viscosity-flow = d_seg-decoupling; "demote from default"). v5/v6 disabled it at 1001; config regressed. FIX: `--l7-start-epoch 1001`.
- **HIGH — directional basis under-allocated.** `--n-dir-freqs 2 --freq-along 4` vs the #1 measured lever (all-class directional, −48%) + the measured 3.2× along-tangent deficit (fix was n_dir_freqs 2→4). ~0 byte (free B, rule 118). FIX: `--n-dir-freqs 4 --freq-along 8`.
- **MED — lane seeded via a no-op** (`--lane-prior-phi1-mode replace`); only `--structured-init-include-lane` is live. Class-1 (0.59%/IoU0.263/~19% of flips) is the rare-class-birth blocker. FIX: drop lane-prior-replace + add a `part_frac[lane]>0` post-init assert.
- KEEP: mod-dim 26 (Whitney-adequate for ~8-dim manifold), chroma (measured d_seg lever, correctly ON), softmax-temp 1.0→0.05, w_seg100/w_pose0. Deferred adaptive-ε/paint = CORRECT (not under-powering). tau@300/muon@726 = PR95-echo → event-trigger is the follow-up A/B.
- BINDING (L3): d_seg load-bearing ONLY with liveness stamp + rollback + positive-control sentinel.

## BENCH 2 — Assumption-Adversary / Contrarian / PR95-author (confound-skeptic) — RETURNED
**VERDICT: PROCEED_WITH_REVISIONS** (2 mandatory edits). "16 confounds killed, 2 survived INTO the flag string where gates check defaults not overrides."
- **CHARGE 1 CONFIRMED (VERIFIED-VIA-SOURCE) — verdict-pairs 96 = non-n600.** witness_autoconfig.py:336 ("proven arm --verdict-pairs 96 ... default 24 degraded"), :441 SCORER_FIXED. Gate #401 passed only because it checks the DEFAULT (0), config OVERRIDES to 96. FIX: → 0.
- **CHARGE 2 CONFIRMED (VERIFIED-VIA-SOURCE, highest-leverage) — fixed-β=4.** Independently traced: model β=4 (2436), levelset `_hosc_beta_for_epoch` returns None w/o `--hosc-beta-end` (1982-1985), loop only updates if non-None (5895) ⟹ constant β=4. **RECONCILIATION (decisive): witness_autoconfig.py:549 — annealed-hosc is a SEALED-205 DELTA, NOT part of proven_base. So the runs that descended to d_seg 0.005 used the ANNEALED sealed delta; proven_base fixed-β=4 is NOT the config that produced good numbers** → a weak d_seg would be un-attributable (known activation confound vs real limit). FIX: step_basis or anneal 1→4. [Contrarian FORK — resolved: the argv check IS available (autoconfig:549); fixed-β=4 was never the descending config; edit stands.]
- **CHARGE 3 PARTIAL (low-risk) — lane-prior replace inert;** fresh run so C10 doesn't apply, low-consequence class. FIX (hygiene): drop, or add efficacy assert.
- **CHARGE 4 REFUTED as blocker — reorient-50** self-heals under rollback (1187 self-rearm); watch-item (confirm liveness visible).
- **CHARGE 5 REFUTED — resume: FRESH is CORRECT.** ep100 weights are themselves a product of the poisoned optimization (legacy freeze, eik-domination, uncalibrated units); even w/ warm-start-EMA you'd measure "recovery from a contaminated basin." KEEP fresh.
- **NEW PREFLIGHT GATE (PR95-author):** refuse `--activation hosc` at the levelset entry w/o `--hosc-beta-end` or `# FIXED_BETA_OK:` — extinct the base-vs-levelset safety-default-divergence class permanently.

---

## ═══ SYNTHESIS — T3 VERDICT: PROCEED_WITH_REVISIONS (unanimous 3/3 benches) ═══

**The meta-finding (all 3 converge):** the 18-confound cleanup fixed the CODE, but two confounds rode the explicit CONFIG FLAGS into `proven_base` — where the L2 gates (which check argparse DEFAULTS) can't see them. `proven_base` is confound-cleaned on 16 axes, re-poisoned on 2. "Clean of the session's 18 confounds" ≠ "clean measurement."

**MANDATORY edits (all benches agree):**
1. `--verdict-pairs 96` → **`--verdict-pairs 0`** (n600; kills 2.5× noise + optimizer's-curse selection bias). VERIFIED-VIA-SOURCE.
2. `--hosc-beta 4.0` (fixed) → **`--hosc-beta 1.0 --hosc-beta-end 4.0`** (annealed, the empirically-descending form) OR **`--activation step_basis`**. VERIFIED-VIA-SOURCE that the levelset loop runs constant β=4 (CLAUDE.md-forbidden). The Contrarian fork is RESOLVED (autoconfig:549: proven_base fixed-β≠the descending config).
3. `--l7-start-epoch 600` → **`--l7-start-epoch 1001`** (disable the measured d_seg-decoupling defect).

**HIGH edits (bench 1+3):**
4. `--eikonal-weight 0.01` → **`0`** for the BASELINE (scorer-invisible off-boundary regularizer; the term that froze both runs). Ballé DISSENT (recorded): run eik-off baseline AND eik-0.01 same-seed A/B; the R-roundtrip d_seg gap arbitrates — no load-bearing eikonal verdict from one arm.
5. `--n-dir-freqs 2 → 4`; `--freq-along 4 → 8` (the #1 measured lever; 3.2× along-tangent deficit; ~0 byte).
6. Drop `--lane-prior-phi1 --lane-prior-phi1-mode replace` (measured no-op); keep `--structured-init-include-lane` + add `part_frac[lane]>0` assert.

**KEEP (unanimous):** FRESH-from-scratch NOT resume · rollback default · C3 eik-norm · per-group-clip · closed-loop liveness · chroma · w_pose=0 (label d_seg-COMPONENT, no pointer claim) · mod-dim 26 · softmax-temp anneal · muon@726 · length-weight 0.001.

**ANSWERS to the main agent's questions:** Q1 config → FRESH proven_base+edits (my resume-ep100 rec OVERTURNED). Q2 eik → OFF in baseline. Q3 adaptive-ε/paint deferral → CORRECT. Q4 resume-valid → NO (C10 voids init levers=FAKE + C8/C9 + poisoned-basin weights). Q5 verdict-96 → MUST be 0. Q6 hosc-β → MUST anneal/step_basis (verified constant-4). Q7 lane-replace → drop. Q8 → Shannon's pre-registered gate (apparatus-precondition every-epoch + d_seg ep0/ep50≤0.9×/ep100 PRIMARY GO≤0.008 ABORT>0.015 /ep300≤0.005 /PROMOTE first-ckpt≤0.004→byte-close+exact-eval).

**FOLLOW-UP BUG-CLASS FIX (per "bugs permanently fixed + self-protected"):** the base-vs-levelset anneal-default divergence (base auto-anneals hosc, levelset silently doesn't) → new preflight gate refusing levelset `--activation hosc` w/o `--hosc-beta-end`/waiver.

**RECURSIVE SELF-REFLECTION (Catalog #363, round 2):** the symposium's own load-bearing claims + their empirical-verification-status: (a) verdict-96-is-non-n600 = VERIFIED-VIA-SOURCE (autoconfig:336). (b) fixed-β=4-runs-constant = VERIFIED-VIA-SOURCE (two benches independently traced levelset:1982/2436/5895). (c) proven_base-fixed-β≠descending-config = VERIFIED-VIA-SOURCE (autoconfig:549) — this RESOLVES the Contrarian fork (no un-verified assumption remains). (d) eik-off-is-optimal-baseline = INFERRED (score-orthogonality is sound but the eik-off vs eik-0.01 arbitration is the pre-registered A/B, not yet measured) → verdict edit #4 is PROVISIONAL-pending the A/B, correctly framed as baseline+arm not a kill. No verdict rests on an ASSUMED-unverified claim. SEAL.

**Mission-contribution: frontier_protecting** (this review PREVENTED launching a re-poisoned run that would have produced an un-attributable d_seg — the exact session-poisoning failure the confound work extincts).
