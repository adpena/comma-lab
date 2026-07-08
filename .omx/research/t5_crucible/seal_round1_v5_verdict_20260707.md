---
doc_type: t5_crucible_seal_round1_v5_verdict
role: SEAL ROUND 1 of 3 (both lenses: LENS A recursive adversarial verify + LENS B deep-math
  meat hunt) on DRAFT v5 — counter was reset 0/3 by round-2 findings on v3; v4 folded those 13;
  v5 folded the 10 CT items (+2 gated extras). This round re-verifies BOTH the folds and the
  regressions.
date: 2026-07-07
target: DRAFT_OPTIMAL_STACK_v5_20260707.md (b241cf466)
verdict: CLEAN (counter 1/3) — zero BLOCKER, zero MAJOR; 9 MINOR/nits listed (incl. the
  mid-round requirement-R scope audit's 3), none of which changes a decision, a load-bearing
  number, or a build item (reasoning printed per item; all BIND to the P7 assembly / next
  editorial fold)
operator_pins_folded: requirement R (verdict-scope taxonomy INSTANCE < FORMULATION < FAMILY <
  PARADIGM) arrived DURING this round — §3b below is the full scope-inflation audit of every
  negative verdict v5 consumes, graded against the pin's BLOCKER bar.
axis: all numbers [macOS-CPU/MLX advisory] unless tagged; pointer contest-CPU 0.19110 UNMOVED —
  this verdict is MEANS.
review_status: fresh-eyes-reviewed(1) — this verifier authored none of v1..v5, the CT seats,
  the recess wave, or the prior verdicts; every load-bearing number below RE-EXECUTED
  ([re-executed], .venv python at full precision / live grep of the trainer argparse + DSL)
  or re-read against the primary on-disk artifact ([verified-by-inspection]).
---

STORES CONSULTED: ORCHESTRATION_LEDGER.md (full 772 lines — reqs A–P + operator pins J/K/L/M/N/P
as audit angles; the v5/CT-1/CT-2/v4 landing folds = the fold spec) · DRAFT_OPTIMAL_STACK_v5
(full) · DRAFT_OPTIMAL_STACK_v4 (full — the inherited base; §0.1 13-item table, §0.2, §1.2,
§2.2e, §4a/§4c, §5.1, §7b, §10, self-attack) · seal_round2_verdict_20260707.md (full — the 9
regression items + §4 certified list) · ct_deepresearch_1 §1.4/§2.3/§3.2/§3.3/§3.4/§4.2/§7.3/
§8.2/§9.2/§10.2/§11/§12 (fold-faithfulness source) · ct_deepresearch_2 §11/§12/§13 (fold-
faithfulness source) · recess_wave1_R1_R3_R6 (full — R1/R3/R6 measured rows v5 consumes) ·
negatives_scale_validity_review (item structure; items 5/6 cite-checked) · trainer argparse
LIVE RE-GREP this session (248 add_argument flags extracted; every §1.0-named flag checked
one-by-one; defaults/choices inspected for --render-aa/--amplify-persist/--amplify-weight/
--curriculum-plateau-windows/--eikonal-visco-ca-band/--eikonal-visco-margin-factor/
--margin-saliency-tau) · src/tac/witness_dsl/curriculum_dsl.py (PersistenceTopology override
emission L2317-2319; AACoverageRender factory L2180-2207). NOT consulted: durable-state files
(stale per sweep); no training launched; $0 recomputation + grep only.

# SEAL ROUND 1 (v5) — CLEAN. Counter 1/3.

## §1 LENS A — RECURSIVE ADVERSARIAL VERIFY

### 1.1 Crossing arithmetic re-verified END-TO-END, unrounded, independently [re-executed]

Recomputed in fresh python (Decimal/float64, no reference to v5's chain):

- pose term √(10·3e-5) = 0.017320508 → **0.0173205** ✓
- central rate 93,092×25/37,545,489 = 0.061986142 → **0.0619861** ✓ · win9 81,032 →
  0.053955883 → **0.0539559** ✓
- S rows: (0.0011, central) = **0.1997336**, over by 0.0086336 = **4.850×** margin ✓ ·
  (0.0010, central) = **0.1897336**, margin **0.0013664** ✓ · (0.0010, win9) = **0.1817034**,
  margin **0.0093966** ✓ · (0.0011, win9) = 0.1917034, over by 0.0006034 ✓
- required targets: central **0.0010137** ✓ · win9 **0.0010940** ✓
- ILC bar 0.0011 − 1.0427e-4 = **9.9573e-4** ✓ (consistency row (c): 0.4% from 0.0010 ✓)
- ν = ln(3.3e-3/2.4e-4)/100 = 0.0262104 → **0.026210** ✓ · s\* = ν·5.4e-4 = 1.415361e-5 →
  **1.4154e-5** ✓ (5 s.f. of the unrounded product; the printed "0.026210 × 5.4e-4" is the
  rounded display of an unrounded chain — req-J compliant) · Δt = ln(6.8e-5/s\*)/ν = 59.88 →
  **~60 ep** ✓ · settle 3/ν = 114.46 → **≈115** ✓ · dwell ln(1.275)/ν = 9.27 → **9.3 ep**,
  250/9.3 = **27×** ✓
- consistency rows: (a) 0.062·ln5 = **0.0997852** vs measured 0.10 → 0.21% ≈ "0.2%" ✓ ·
  (b) 0.00178/25 = **7.12e-5** vs 5e-3·0.0034/25·100 = **6.8e-5** → ratio 1.047, "within 5%" ✓
- λ_bytes = 25/37,545,489 = **6.65859e-7** ✓ · hood 8 B = **5.32688e-6 S** ✓ (round-2's
  MINOR-R2-5 fix holds exactly)
- M1 unit correction (§0.0b): 0.3×0.00178 = 5.34e-4 S = **5.34e-6 d_seg** ✓; 1× margin =
  **1.78e-5 d_seg** ✓ (P-DZ bands match §0.0b — the CT-2 unit mix was corrected, no number moved)
- M1 deadzone floor: (1/255)/(0.842·g_I) at g_I ∈ [0.2, 0.5] = **[0.0093, 0.0233] px** ✓
- EWMA/ILC: |1−0.7ξ| at ξ ∈ [0.65,1.35] = **[0.055, 0.545]** ✓; ω = 0.5 stable ⟺ ξ ∈ (0,4) ✓
- EVSI pose row: √(10·0.0011) = 0.1049 ≈ 0.105; swing 0.0876 ≈ **0.088**; min(p,1−p)·swing =
  **0.044** ✓; table sum ≈ 0.0485 + 3e-4·k → **"0.048–0.05"** ✓
- r\* plug: 0.674·√2 = **0.9532** → "0.95·σ_eff"; ×1.5 = **1.4298 ≈ 1.43 px** = the measured
  knee ✓ · Conley threshold 0.0998 + 0 ≈ **0.10 logit** ✓
- cadence plugs: 0.00178/3.3e-3 = 0.54 → floor 25 binds ✓; /2.4e-4 = 7.4 → floor ✓;
  /1.4e-5 = 127 → cap 100 binds ✓
- TAIL budget (audit angle 6): 115 + 150 = **265 ep/cycle** ✓; 265×7 = 1,855 ≤ 2,350
  (= 3,000 − ep650 entry) ✓; at the 350-ep upper, 2,350/350 = 6.7 ≈ 7 — k_max "3–7"
  (τ\*-limited lower, budget-limited upper) internally consistent ✓
- §8.5 ES sizing: 2×0.00178/λ_bytes = 5,346 B ≈ **5.3 KB** ✓; 2×0.015/λ_bytes = 45,055 ≈
  **45 KB** ✓ · win9 saves 12,060·λ_bytes = **0.008031 S = 4.51×** ✓ · g_dec 100×1.0427e-4 =
  **0.010427 S = 5.86×** ✓ · SC-7 bands 0.5×/2× margin = 8.9e-4 / 3.56e-3 S ✓
- win9 display-artifact note verified: rounded column entries sum 0.181704 vs unrounded
  0.1817034 — v5's explanation is exactly right ✓

**Every load-bearing number in the crossing chain, the fold laws, and the consistency rows
reproduces to the printed digit.** The g_dec selection logic (decoded-KKT argmin over
checkpoint × quant-depth × section options, F13-supplied) is coherent and inherited unchanged.

### 1.2 Flag-reality audit (angle 2) [re-executed — full 248-flag argparse extraction]

37 of 39 §1.0 "Verified EXISTING" spellings EXIST (incl. all six --margin-saliency-\*, all four
--eikonal-visco-\*, both --stage-transition-rewarmup legs, --amplify-margin-target). The two CT
catches re-verified as genuinely ABSENT (`--copred-verdict-window`, `--island-dilation-radius-end`
— both MISSING from argparse: real catches, correctly routed to B1/B18). `--tau-fin-slope-star` /
`--verdict-cadence-law` correctly held as PROPOSED-NEW, never claimed existing. Choices verified:
`--render-aa {none,supersample,ipe}` ✓; `--amplify-persist {uniform,inverse_thickness}` ✓
(persistence_pairs correctly a B13 build, not claimed). **Two spelling defects found → MINOR-A1
below.** Line refs in §1.0 drifted ~11–22 lines vs current HEAD (file grew post-v5) — cosmetic,
flags all present.

### 1.3 Fold-faithfulness vs CT-1/CT-2 (no drifted number, no dropped gating, no resurrected DEAD)

[verified-by-inspection against both CT sources] Fold 1: s\*/ν/60-ep/42-min/would-fire-first/
P-CT3 band+kill/cap-726 all verbatim-faithful; v5 ADDS the M2-class guard (ep685 > anneal-complete
ep600) — an improvement, not drift ✓. Fold 2: bar/ω/γ/contraction/2–3-runs faithful; crossing
ceilings unchanged ✓. Fold 3: V=4→5 faithful with the invented-flag correction; PMP row faithful
with the operating-point caveat carried ✓; turnpike 265/3–7/dwell-115 faithful ✓. Fold 4: Conley
law faithful; v5 ADDS the ≥95% band above CT-2's kill-only spec (tightening) ✓; sufficient-not-
necessary + smoothed-field caveats carried ✓. Fold 5: signed-hinge law, Q1 gate thresholds, 2.2×
efficacy bound, through-R smearing note all faithful ✓. Fold 6: 9-row c(τ) + r\* faithful ✓.
Fold 7: union dedupe EXACT — 13 + (5+5) − 4 absorptions (2→SC-7, 1→SC-3, 2→one new SC-16) =
**19 rows, 6 NEW + 3 EXTENDED** ✓; all CT-2 §12.2 five signals present (SC-7ext/16/17/18/19) ✓;
all CT-1 §12.2 five present (SC-14/SC-7/SC-3/SC-15/SC-16) ✓. Fold 8: five bounds + BLR non-bound
faithful; unit fix correct ✓. Fold 9: DEAD ledger complete (backstepping/LQR/HJB/ES-dither/
Griewank/Hajek-cooling/Imbert–Monneau); Hajek-cooling-DEAD vs Hajek-M4-bound are different
objects (schedule import vs impossibility bound) — no contradiction ✓; max-plus essence in §11
row 16 ✓. Fold 10: see 1.1/1.2. Extras +A/+B: both ARE §12.1 IMPORT-NOW members (CT-1 §12.1(4),
CT-2 §12.1(5)) — charter-clean; both gated (P-CT2; injection+A/B) with fail-safes ✓. Both CT
§12.1 sets verified 5/5 folded each.

### 1.4 Gating coherence B16/B17/B18/B-CT1 (angle 5)

All four resolve middle-zone outcomes to the SAFE default state (B16: 0.1≤|ρ|<0.3 → stays
default-off in the duty queue; P-CT3: fire in [650,670)∪(700,726] → not-band → arm stays
would-fire-only; P-CON 80–95% → not-passed → advisory ships with safety-factor path; B18 →
fixed-ramp fail-safe). Promotion-requires-band + kill-retires + else-default is well-formed —
the chain-A kill-middle-zone lesson is honored by construction. ✓

### 1.5 Regression check — every round-2 finding stays fixed in v5

| round-2 finding | v5 state | verdict |
|---|---|---|
| BLOCKER-R2-1 AA(ss=2) refused; R1 bytes | §1.1 `AACoverageRender(mode="ipe")`; §5 LBND2 41,526 / win9 18,832 / win5 QUARANTINED; ss=2 in §9.4-inherited with measured cost | **FIXED** ✓ |
| MAJOR-R2-2 stale 0.011 | §2.3 "ratio 0.163 K=128; conjunction-dispositive"; I-6 rewritten (v4 §10, inherited); I-7 rows all PROVISIONAL-tagged | **FIXED** ✓ |
| MAJOR-R2-3 per-class weights unbuildable | B13 build item + pooled-1.0 fail-safe written; §1.0 lists B13 flags as PROPOSED-NEW | **FIXED** ✓ |
| MINOR-R2-4 `--tau-anneal-end` | `--softmax-temp-end` everywhere; grep-verified EXISTS (default 0.05) | **FIXED** ✓ |
| MINOR-R2-5 hood 10× slip | 5.32688e-6 S — recomputed exact ✓ | **FIXED** ✓ |
| MINOR-R2-6 adaptive-ε stale sentence | τ-coupled row + saturation ALARM (§4 rank-4; §1.4 row 1) | **FIXED** ✓ |
| MINOR-R2-7 recon-gap unranked | §4a rank-1 self-orient persist 8.10× NAMED LB | **FIXED** ✓ |
| MINOR-R2-8 AA×island row + along=8 sentence | §4a rank-5 row + §0.3 regime sentence (v4, inherited; §1.4-end names it) | **FIXED** ✓ |
| MINOR-R2-9 reactive-only laws; RS unnamed | §2.2f quantitative feedforward arm; ε_ff term; SC-11 RS-1..5 persisted DECIDE models | **FIXED** ✓ |

Lane-anisotropy scope pin: §2.3 carries the u_min-isotropic scope sentence; no anisotropic lever
(B16/Rebalance/comb/band) is demoted on that negative ✓.

## §2 PROVENANCE AUDIT (L81 — every load-bearing measured claim)

| claim | anchor | review status | form limitations (carried in v5?) |
|---|---|---|---|
| g_dec = +1.0427e-4 | recess R6: r6_verdict_pairs.jsonl + packet 20260708T013253Z (0.0036146 − 0.0035103 recomputed ✓) | self-executed-pre-registered (bands pre-registered in P5 §4; executor-flagged; deterministic re-run path on disk) | ONE checkpoint (mod32cap ep650), pose-blind — v5 states both; per-stage SC-7 re-measures ✓ |
| win9 18,832 B / LBND2 41,526 B | recess R1 artifact JSON (lbnd4_on_smoothed_r1_measured.json) | pre-registered band, PASS; win5 roundtrip-FALSE quarantined | lossy-geometry trained-with leg OPEN (P8/F8 gate) — carried ✓ |
| transition forfeit 5.4e-4 S / 2.7e-3 S | levelset_train_result.json ep600/625/650 | round-2 [re-executed] to 7 dp | trace-exact ✓ |
| m_q = 0.10 flip edge | birth_death_persistence 20260630 L134/L196 | round-2 [re-executed] | GT-margin bins; τ-law re-derived per stage ✓ |
| ν = 0.026210 (→ s\*, 115, 265, 9.3, 60) | mod32cap trace ep350→450 slopes, derived by CT-1 | **fresh-research-round-1-UNREVIEWED** | MITIGATED: labeled DERIVED-from-one-trace (self-attack 2); P-CT1 $0 refit queued FIRST with kill-recomputes-all; arm ships would-fire-only. The one spec-NOW consumer is V=4→5 — conservative direction (later exit), P-CT1-covered. NOT a finding |
| Muon quench +27.5% / μ = 1.275 | S2 M1 measured row | prior-round verified | ✓ |
| dilation knee 44.6/90.0/98.3% @ σ=1.5 | 0630 island probe (via CT-2) | measured anchor, plug-consistency re-verified (0.953×1.5 = 1.43) | ✓ |
| \|H_R\| ∈ [0.842, 1.0] all-pass | negatives review item 5 | SCALE-ROBUST (the rare fully-surviving negative) | uint8 nonlinearity caveat lives in the source ✓ |
| chain-A ratio 0.163 / extrap 0.08 | terminus commit 42fa00812 | round-2 adjudicated (conjunction-dispositive) | ✓ carried verbatim |
| Δ_dec^logit | — | honestly UNMEASURED, init 0, supplied at first byte-close | stated ✓ |
| ALL CT-derived laws (s\*, ILC, PMP, Conley, r\*, signed-hinge, cadence, damper) | CT-1/CT-2 | **fresh-research-round-1-UNREVIEWED, tagged on every row** | every behavior-changing use is would-fire-only OR $0-probe-gated OR fail-safed OR telemetry; I-7 registrations all PROVISIONAL — L81-compliant. No v5 decision rests SOLELY on an unreviewed CT derivation without a measured anchor or queued probe ✓ |

The three §0.4 consistency rows (independent derivations hitting measured constants at 0.2–5%)
were each re-computed here and hold — they are genuine evidence the CT derivations sit on real
structure, and v5 correctly registers them PROVISIONAL rather than leaning on them.

## §3 LENS B — DEEP-MATH MEAT HUNT (findings + what survived)

Hunted: the full CT adoption lists vs v5's folds; the c(τ) enumeration's own audit rule turned
on the live flag set; synergy seams of the four new levers; the impossibility-bound → signal
mapping; naive-collapse shapes in the new laws.

**Survived (certified, do not re-litigate in rounds 2–3):** the forfeit-matched law's fixed-point
structure (forfeit shrinks ⇒ s\* shrinks ⇒ fire later) is correct control theory and the
restore-law interaction is coherent (fire@685 ⇒ window contains ep650 ⇒ forfeit→0) · the
zero-DC-gain argument for B-CT5 is structurally sound (high-pass cannot bias the converged
score) · the two-semiring split (M5) and the deadzone floor arithmetic (M1) check · the union
ledger has zero write-only rows (every SC row's consumer exists or has a §10 build) · verdict-
budget self-attack arithmetic honest (+12% worst case, −30–40% if P-CT2 passes) · B15's
condition VERIFIED TRUE this round: the DSL AACoverageRender factory hardcodes
`"--render-aa": "supersample"` (curriculum_dsl L2198) ⇒ B15 is confirmed-needed, exactly as the
build row anticipated · impossibility-bound mapping M1→SC-16/P-DZ, M3→SC-18, M4→SC-10, M5→SC-19/
P-MP all land in the ledger; M2 rides Q2/F12 (default-ON telemetry per v4 §7b) — mapped, though
carrying the F12 label overload noted below.

### Findings (all MINOR; per-item bar test printed)

**[MINOR-A1] (lens A, angle 2) Two flag spellings in §1.0's Verified-EXISTING list do not
exist.** The shorthand `--persistence-loss-weight/-warmup-epochs/-classes` expands to
`--persistence-loss-warmup-epochs` / `--persistence-loss-classes`; the real flags are
**`--persistence-warmup-epochs`** and **`--persistence-classes`** (argparse L7879-region;
--persistence-loss-weight itself is real). Same class as MINOR-R2-4. Consequence check: the
launch path is DSL-compiled and PersistenceTopology's factory emits the TRUE spellings
(curriculum_dsl L2317-2319 verified) — no launch artifact, decision, number, or build item
changes; §0.1 row 10's "zero invented flags remain" needs a 2-token amendment. Stated plainly:
under round-2's grading convention this would have been logged the same way MINOR-R2-4 was; under
this round's operator-refined bar (decision/number/build-item) it is a doc fix. BINDS to P7/v6.

**[MINOR-A2] §0.0a M3 row carries an unpinned "> x%" threshold** (inherited verbatim from CT-2
§13). A deciding-measurement band with a symbolic threshold is TBD-class under the P2 contract's
letter. Mitigations: asymptote-layer only (no run-1 knob or crossing number consumes x); SC-18 +
Q3 generate the underlying data regardless; the run-2 asymptote computation can pin x from the
measured clamp-binding distribution. Changes no run-1 decision/number/build item. BINDS to P7/v6
(pin x or declare it computed-from-SC-18).

**[MINOR-B1] CT-1 §11 rank-6 (B-CT2: evaluate the adaptive-ε law at τ(t+H), ~5 LOC, gated on the
same Q3 clamp-binding probe as the imported rank-7 B-CT4) has no disposition row in v5.** Not a
fold-contract violation (B-CT2 is in neither the 10-item fold list nor either §12 IMPORT-NOW
set), and v4's inherited ε_ff(t) = ε(ĉ_a(τ(t))) already embodies the τ-schedule-as-internal-model
mechanism — the residual delta is only the one-horizon-ahead evaluation, whose benefit CT-1
itself left "unquantified". One recorded line owed (subsumed-by-ε_ff OR queued-behind-Q3).
Changes no decision or number; the build list's content is not forced either way.

**[MINOR-B2] B16 × AmplifyIsland seam undeclared.** Both ride the SHARED LEVER-4 signed-margin
machinery (the amplify flag's own help: "rides the SHARED LEVER-4 _signed margin"); if Q1 fires
B16 while the island hinge is active, lane-boundary pixels carry two margin-band weights whose
composition law is unstated (req-I grain). Mitigations: B16 is default-OFF + Q1-gated (run-1
primary unaffected); its pre-registered A/B is vs the unsigned margin-gate, which partially
covers the seam. One coupling sentence owed.

**[MINOR-B3] `--eikonal-visco-ca-band` (the c_a measurement window on the margin field; default
0.0 = interior mean) is τ-adjacent and sits outside the 9-row c(τ) enumeration.** The
enumeration's own audit rule ("any constant whose consumer reads the τ-smoothed margin field…")
reaches it: interior-mean vs annulus-restricted c_a diverge exactly as τ descends (the memo's own
flat-regime coupling). Mitigations: default 0.0 is the symposium exact launch formula
(measured-anchored at coarse τ); row-1's floor law + the saturation ALARM + SC-18 instrument the
consequence; F12 re-validation is the natural cover. A 10th enumeration row (declared
mode/exponent or measured-flat verdict) owed. No run-1 number moves.

### §3b REQUIREMENT-R SCOPE-INFLATION AUDIT (operator pin, arrived mid-round; every negative verdict v5 consumes, graded INSTANCE < FORMULATION < FAMILY < PARADIGM)

**The four known prior over-scopings — all correctly DE-INFLATED in v5 [verified-by-inspection]:**
viscosity → REOPENED, run-1 eikonal ramp = first fair test (v4 §7b Q3, inherited) ✓ · UniWARD
pooled null → explicitly read as the sign-integrated FORMULATION ("this law with the sign field
integrated out — the theorem PREDICTING the measured null"), Q1 re-tests per-side ✓ · mod-dim →
held OPEN ("mod-dim 2-point named run-2", SC-11 consumer), never cited as refuted ✓ · l7 →
demoted-as-PR95-ordered from the DEFAULT graph only, re-entry behind a restore-guard (§2.3
mode-admission rule) — formulation-scoped demotion, not a kill ✓.

**Per-verdict scope table (BLOCKER bar = a one-formulation kill treated as family truth):**

| negative v5 consumes | v5's operative scope | inflated? |
|---|---|---|
| chain-A TerminalSolve NO-GO / "exhausted both orders" | INSTANCE/basin-scoped by its own words ("at this basin", "at the frozen schedule point"; K≥32+K-trend SENSOR adjudicates any future basin; §2.3(4) forbids citing "35%" at a τ=0.062 ckpt) | NO ✓ |
| AA supersample ss=2 (REFUSE + −49% harm) | FORMULATION-scoped kill (ss=2/ndf-4/full on THIS box); AA family survives as ipe; re-entry condition written (hardware change ∧ paired verdict) | NO ✓ |
| win5 roundtrip-FALSE | INSTANCE quarantine "until the identity defect is explained" | NO ✓ |
| u_min-isotropic negative | scoped lane-dilute + coarse-τ; the lane-anisotropy pin honored (no anisotropic lever demoted on it) | NO ✓ |
| PR95-L25 temporal-delta +64% | measured ON the object it would ship on (our code stream) → instance decision applied as instance DROP | NO ✓ |
| #207 pre-emphasis/deconv DEAD | legitimately FAMILY-level: an OPERATOR property (\|H_R\| ≥ 0.842 all-pass bounds the gain of ANY deconvolution form) — negatives item 5 SCALE-ROBUST with the uint8 caveat carried in the source | NO ✓ (family scope EARNED by derivation covering the formulation space) |
| hosc fixed-β divergence | consumed only as the annealed-hosc design response (the scoped fix) | NO ✓ |
| Hajek log-slow cooling "already REFUTED" (§12) | pointer-cite; the REFUTED object is the schedule-IMPORT formulation — the theorem itself SURVIVES in v5 as the M4 impossibility bound | wording owed → R-2 below |
| backstepping "DERIVED-DEAD for this plant" (§12) | row subject = "kernel machinery (Volterra/Goursat)" = formulation-named; leg (a) no-1-D-causality kills the kernel-transformation FORM; legs (b) actuator-through-loss-not-boundary + (c) descent-already-Lyapunov-stable are form-INDEPENDENT for this plant ⇒ the family-for-this-plant scope is covered by (b)+(c), and the family's transform-to-solvable-coordinates ESSENCE is retained (§11 row 16) | NO, but the row should say which leg carries family scope → R-2 |
| LQR/HJB/ES-dither DEAD (§12) | LQR/HJB: derivation-based family-for-this-plant (no linear plant; dim) ✓; ES-dither: FORMULATION-scoped by its own words (per-step dither on θ; campaign-timescale FD-ES survives) ✓ | NO ✓ |
| B16 Q1 kill ("\|ρ\| < 0.1 both sides ⇒ SCALE-ROBUST dead, retired") | HAZARD: §1.3's sentence, read alone, licenses a scale-ROBUST family retirement from ONE schedule point; the cited normative spec (v4 §7b Q1, inherited) contains the fine-τ-checkpoint re-run, so the operative probe is two-point — no operative inflation, but the SCALE-ROBUST-dead label must be worded as earned only AFTER both τ points | → R-1 below |
| §7c probe gates | P-CT3/P-CON/P-CT2/P-DZ all worded as DISPOSITION deciders (stays would-fire / fit safety factor / stays unbuilt / stays DEFER — P-DZ even says "census — no kill") ✓; P-MP's "kill" should be worded as this-K/this-fit-form unadmitted (formulation disposition), not a tropical-layer truth claim | → R-3 below |

**Requirement-R findings (all MINOR; no operative inflation found — the pin's BLOCKER bar is
not crossed anywhere):**

**[MINOR-R-1] §1.3's Q1 retirement clause omits the fine-τ conjunct.** "if |ρ| < 0.1 both sides
every major pair ⇒ SCALE-ROBUST dead" must read "…at BOTH the coarse and the fine-τ checkpoint
(the v4 §7b Q1 protocol) ⇒ SCALE-ROBUST dead"; as written a reader could retire the lever at one
schedule point under a scale-robust label — the exact R class. The normative spec it cites is
correctly two-point, so no decision changes; one clause owed.

**[MINOR-R-2] §12 DEAD/CAMPAIGN rows lack explicit scope-level tags** (the taxonomy postdates
v5). In-words scoping is present and correct for backstepping/LQR/ES-dither/Griewank; owed:
(i) tag each row {formulation-dead | family-dead-for-this-plant | import-dead}; (ii) backstepping
row states that legs (b)+(c) — not the 1-D-causality leg — carry the family scope; (iii) "Hajek
already REFUTED" scoped to the schedule-import formulation (the theorem lives on as M4).

**[MINOR-R-3] P-MP kill wording** → "K ≤ 64/class fails band-level annulus accuracy ⇒ THIS
expansion form (K, per-class quadratic basis, this fit) stays unadmitted; richer tropical forms
re-enter only with their own probe" — disposition, not truth claim. No disposition changes.

**[nit] Three cosmetic items, listed for completeness, no bar test needed:** (i) B1-spec-change
LOC "0 (spec)" though it also amends the out-of-process advisory's V (a few real LOC); (ii) §1.0
argparse line refs drifted ~11–22 lines vs current HEAD (file grew after v5's grep; all flags
present); (iii) §0.0c's "~10× run-1's direct crossing value" gloss is loose — every consistent
reading of the direct-value formula gives MORE than 10× (conservative direction, claim only
strengthens); (iv) "F12" labels both the stage-wall-clock row (SC-2) and the dash-contrast
τ-samples (Q2) across the doc lineage — a disambiguating rename owed eventually.

## §4 VERDICT + COUNTER

**CLEAN.** Zero BLOCKER, zero MAJOR. Nine MINOR/nit items found and listed with exact locations
(A1/A2 lens-A · B1/B2/B3 lens-B · R-1/R-2/R-3 requirement-R scope audit · cosmetic nits) —
per-item bar tests printed above: **none changes a decision, a load-bearing number, or a build
item** (the two wrong flag spellings have correct-spelling twins that the DSL launch path already
emits; the unpinned x%, the B-CT2/seam/ca-band rows, and the three scope-wording clauses are
one-line editorial folds; the requirement-R audit found NO operative scope inflation — v5's
negatives are conspicuously well-scoped, and all four known prior over-scopings are correctly
de-inflated). Stated plainly per the charter: this round found nothing load-bearing; the nits
BIND to the P7 assembly (or any v6 raised for other reasons) and are not grounds to burn a
designer round. The strongest things that survived re-derivation: the crossing chain (every
digit), the forfeit-matched fixed-point law, the 19-row union ledger (dedupe exact, zero
write-only rows), and the three cross-field consistency rows (recomputed, all hold).

**Counter: 1 of 3.** Round 2 should be delta-scoped to: the six items above if a v6 lands; fresh
eyes on the CT sources themselves (their §1–§10 derivations remain round-1-unreviewed — this
round verified v5's FOLDS of them and their plug-consistency with measured anchors, not every
internal derivation step); and any recess-queue result (P-CT1/P-CT3/P-CON/Q1) that lands
mid-window.

Pointer 0.19110 UNMOVED — this verdict is MEANS.
