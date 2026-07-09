# T5 CRUCIBLE-2 — P5 SECOND RED-TEAM (the pre-seal fresh-eyes pass) — 2026-07-09

**Phase:** P5 (SECOND RED-TEAM → SEAL-TO-RECESS). **Targets:** `SYNTHESIS_v2_v752_20260709.md` (90d562a5a)
+ `P4_recess_20260709.md` (4a089be2b) + `P3_redteam_verdict_20260709.md`. **Surface:** `crucible2_v752`.
`[no-triality]` · **$0 · no GPU · no training · run dirs READ-ONLY** · #205 UNTOUCHED. Pointer contest-CPU
**0.19110 UNMOVED** — everything here is `[macOS-MLX/CPU advisory]` NON-PROMOTABLE MEANS.

**Fresh-eyes rule honored.** I did NOT trust the P4 memo. Every FORCES-CHANGE below was **RE-DERIVED from
primary artifact** (MLX Muon source, trainer argparse, trainer loss/alarm loop, resume routing), not
memo-trusted. `review_status: self-executed, fresh-eyes-UNREVIEWED` (P6 treats every disposition here as a
finding-producing round).

**STORES CONSULTED (all re-derived, not trusted):**
`SYNTHESIS_v2_v752` (§0 disposition table · A.1–A.10 · §B compiled config · §C owed 1–13) ·
`P4_recess` (M1–M4) · `P3_redteam_verdict` (A-1 · W-1..W-10 · HELD×10) · ORCHESTRATION_LEDGER
(§178-186 POSE ENGAGEMENT GATE verbatim · §188-203 WALL-CLOCK · §205-211 DUAL-CHAIN) · PRIMARY CODE
re-derived: `.venv/…/mlx/optimizers/optimizers.py` L896-948 (`_zeropower_via_newtonschulz5` + `Muon.apply_single`)
· `experiments/train_witness_realized_through_R_mlx.py` L1879-1887 (`_is_muon_weight`) + L2414-2418
(`MultiOptimizer` routing) · `experiments/train_levelset_witness_realized_through_R_mlx.py` (argparse
L9977/10213/10242/10307-10318/10328-10335/11036/11270-11293/11425-11470/10706-10762/11125/11144, term_domination
alarm L9222-9236 + constants L8012-8013 + `_reg_keys` L9224, island_amplify loss-term L5065) ·
`levelset_pose_gate_l7best_20260701.partial.jsonl` (σ_min anchor, via P3/P4) · `delta_R_noise_floor.json`
(δ_R=0.0196, via P3). **Not a store: no new measurement / GPU / training taken; run dirs read-only.**

**verdict_scope discipline:** every negative carries `verdict_scope`; every magnitude call carries a cite.

---

## HEADLINE (answer-first)

**VERDICT: SEAL-READY-WITH-AMENDMENTS (6, amendment-grade). NOT a rewrite.** The launch DIRECTION survives.
All three P4 FORCES-CHANGE are RE-DERIVED-CONFIRMED by me from primary artifact (not memo-trusted). The
flag ON-set cross-checks CLEAN against the live argparse (one never-invent-flags footnote). But my **sharpest
new attack — the one the chain has not made — is that A-1's own FIX inherits A-1's never-fire failure mode**:
the RELATIVE relaxation-fit PRIMARY criterion is, AS SPECIFIED (exp-asymptote fit), **not reliably measurable
on the runs we will actually launch** (M1 proves the only σ_min trajectory we have is a CV-0.21 oscillation,
and the fresh run uses the identical k=32 probe). So "ships pose-conditioned-OR-banked-R1" is, as-written,
~always "banked-R1" until the plateau statistic is made robust + de-noised. This is AMENDMENT-grade (the
ship-banked-R1 fallback keeps the launch SAFE regardless), but it means the load-bearing A-1 fix is
insufficient as-written and owed-1 must be re-specified before it can honestly claim "pose-conditioned."

| # | amendment | grade | verdict_scope |
|---|---|---|---|
| **1** | owed-1 PRIMARY: REPLACE exp-asymptote-fit with rolling-slope-≈0 on DE-NOISED σ_min + noise guard | load-bearing (my sharpest attack) | FORMULATION |
| **2** | owed-1 canary: $0 = NEGATIVE-control + SYNTHETIC-positive; REAL positive = governed telemetry-replay (owed, non-blocking) | load-bearing | N/A (feasibility) |
| **3** | owed-1 telemetry: `--jacobian-basin-telemetry` verified default-True (keep, forbid `--no-`); DENSIFY terminal-window σ_min cadence | config | N/A |
| **4** | owed-3: DEMOTE crush-arm (RE-DERIVE-CONFIRMED); the clip is ~magnitude-redundant on BOTH Muon AND Adam (out.*→Adam verified) → Muon-only exemption is redundant+asymmetric; real sensor = per-class d_seg-slope + effective-STEP + direction-cosine | design | FORMULATION |
| **5** | owed-12: alarm ALREADY BUILT (RE-DERIVE-CONFIRMED L9222); EXTEND whitelist to Class-B at SAME 40% + non-halting >25% WATCH band + small-denominator guard; island_amplify 28% = WATCH not lower-firing-threshold | correction | FORMULATION |
| **6** | flag footnote: `--muon-start-event` accepts ONLY `powerlaw_meat`; §B `<nucleation>` / A.5 "tau nucleation" must resolve to it (never-invent-flags) | never-invent-flags | INSTANCE |

---

## PART 1 — RE-DERIVATION OF THE THREE P4 FORCES-CHANGE (fresh eyes; I did not trust the memo)

### 1.1 M3 (amber×Muon crush) — RE-DERIVE-CONFIRMED, and EXTENDED beyond P4

I read the MLX Muon source directly (`.venv/…/mlx/optimizers/optimizers.py`):
- **L906** `X = X / (mx.linalg.norm(X, keepdims=True) + 1e-7)` — Frobenius-normalize, **discards magnitude**.
- **L911-913** Newton-Schulz → singular values ≈ 1 (orthogonalize).
- **L946** `lr *= max(1, update.shape[-2] / update.shape[-1]) ** 0.5`; **L948** `return parameter - lr * update`.

⇒ the Muon update magnitude = `lr · max(1, aspect)^0.5`, **INDEPENDENT of the input gradient magnitude**. A
scalar per-group clip `c≤1` on a Muon param's gradient is discarded by the Frobenius normalization. **The
"double-normalize CRUSHES the seg update" mechanism is code-DISPROVEN.** P4 M3 is correct. The launch-BLOCKING
"assert Muon's seg-update magnitude is not crushed" arm (A.6 / §C owed-3) tests a mechanism that **cannot
occur** — it measures a near-constant and always "passes," giving FALSE ASSURANCE. `# MAGNITUDE_DISMISSAL_OK:
MLX Muon L906 Frobenius-normalize + L946 aspect-only lr; gradient magnitude discarded.`

**MY NEW EXTENSION (the chain has not made this — I re-derived the routing):** `_is_muon_weight`
(`train_witness…mlx.py` L1879-1887) routes **only decoder-hidden weight MATRICES** to Muon; **`out.*` (the
final/output layer — exactly the #341 head-solve target: out_sdf, out_tex, the boundary-sharpening params) →
Adam** (L1884 `if path.startswith("out."): return False`), plus all biases + code-latent → Adam. Two consequences:

1. The per-group scalar clip is **~magnitude-redundant on BOTH optimizer routes, not just Muon.** Adam's
   update is `lr·m/(√v+eps)`; a scalar pre-clip `c` scales m by `c` and v by `c²`, so the update ≈
   `lr·(c·m)/(c·√v + eps)` → the `c` CANCELS in the ratio except near the eps floor. So Adam, like Muon, is
   approximately **magnitude-invariant to a scalar per-group clip.** The whole "double-normalize" family is
   even more disproven: the per-group clip is ~a magnitude no-op on the ENTIRE decoder.
2. Therefore a **Muon-ONLY exemption is redundant (for magnitude) AND asymmetric** — it leaves the equally
   redundant clip on the Adam-routed seg output head (the params that actually sharpen the boundary). The
   design-fix as-scoped ("exempt Muon") does not match the true (optimizer-agnostic) mechanism.

**The genuine residual (survives, and is optimizer-agnostic):** the per-group clip fires INTERMITTENTLY
(`c<1` only when the group norm exceeds grad_clip — early/blowup epochs yes, converged no). Under intermittent
firing, `c` does NOT factor out of the momentum buffer, so the orthogonalized/normalized DIRECTION drifts vs
no-clip — on BOTH Muon's `v` and Adam's `m,v`. This is a DIRECTION effect, not a magnitude crush. `verdict_scope:
FORMULATION` (the "double-normalize crush" mechanism + the Muon-only, magnitude-assertion arm; the exemption's
intent and the observability concern survive, re-grounded to direction + effective-step).

### 1.2 M4 (term_domination already built + whitelist gap) — RE-DERIVE-CONFIRMED

I read the trainer loop directly: **L9222-9236** the `(C6) term_domination alarm` fires when the max
**regularizer** term fraction `> _TERMDOM_FRAC (0.40, L8012)` for `≥ _TERMDOM_MIN_ROWS (3, L8013)` sustained
rows; the whitelist is **`_reg_keys = ("eikonal","length","eik_steik","boundary_distance")` (L9224)**.
**island_amplify IS a loss term** (`terms_out["island_amplify"] = _amp_contrib`, L5065; key-list L2855) and is
**NOT in the whitelist.** ⇒ W-3 / owed-12 "not implemented, build it" is FACTUALLY WRONG (the alarm exists;
both prior audits grepped `confound_gates.py` and missed the trainer loop). P4 M4 is correct. The synthesis's
paraphrase "any single loss term > 40%" would false-fire 100% on the primary `seg` term — the "regularizer /
exclude the primary objective" qualifier is LOAD-BEARING. `verdict_scope: FORMULATION` (the alarm's
term-coverage + the paraphrase; the alarm mechanism is CONFIRMED sound).

### 1.3 M1 + M2 (the σ_min sensor + canary) — accepted as the recess measured them; see PART 3 (my sharpest attack builds on M1).

---

## PART 2 — THE v3 AMENDMENT SPEC (amendment-grade; folds the FORCES-CHANGE)

### AMENDMENT 4 — owed-3 (amber×Muon): demote to WATCH; re-scope the design-fix + the sensor

- **DEMOTE** the "assert Muon seg-update magnitude not crushed" arm from LAUNCH-BLOCKING → **WATCH** (M3: it
  tests a non-mechanism; leaving it launch-blocking spends a launch-gate on a guaranteed-pass).
- **The design-fix re-scope (my extension):** a Muon-only per-group-clip exemption is redundant-for-magnitude
  AND asymmetric (out.*→Adam). Two admissible re-scopes: **(i)** DROP the per-group scalar clip entirely (it is
  ~magnitude-redundant on both normalized routes; the global `clip_grad_norm` already bounds the blowup), OR
  **(ii)** keep it but apply it SYMMETRICALLY and measure the **update-DIRECTION cosine (clip-on vs clip-off)**
  on BOTH Muon and Adam groups + the **effective-STEP** `‖lr·update‖` (not gradient magnitude). Either lands
  the actual mechanism; the Muon-only magnitude exemption does not.
- **"What sensor DOES catch magnitude-flat Muon seg suppression?" (the P5 question):** because the Muon update
  is ALWAYS `lr·√aspect` regardless of gradient, `gnorm_hijack`/spike-guard (magnitude alarms) are
  STRUCTURALLY BLIND to it. The suppression-catching sensor is NOT a magnitude alarm — it is **(a) a per-class
  d_seg descent-SLOPE monitor vs an amber-OFF reference at equal epochs** (the v2 A.6 "slope flattens earlier"
  signature — this IS the right test, but its RATIONALE is direction/effective-step, not a magnitude crush),
  plus **(b) effective-step telemetry `‖lr·update‖` per param group**, plus **(c) direction-cosine under
  intermittent clipping**. This arm belongs in the FIRST joint amber arm as a WATCH, not a launch-blocker.
  `verdict_scope: FORMULATION`. Re-formulation queue already satisfied by (a)/(b)/(c).

### AMENDMENT 5 — owed-12 (term_domination): correct + extend + do NOT lower the firing threshold

- **CORRECT the synthesis:** owed-12 is NOT "build the alarm" — the alarm is BUILT (L9222). The D1
  compose-safety runtime leg is CODE-BACKED for the 4 whitelisted regularizers (W-3's "config comments only"
  is wrong). Re-scope owed-12 to **EXTEND the reg-whitelist** to the un-guarded Class-B loss terms
  `{temporal_screw, area_constraint, island_amplify, chroma_boundary, margin-hinge}` at the **SAME 40% firing
  threshold**.
- **island_amplify at 28% — worth a LOWER per-lever firing threshold? NO (I reject that branch).** island_amplify
  is a BIRTH-AMPLIFICATION force whose 28% during its active birth window is plausibly **BY DESIGN** — it must
  be strong to nucleate islands against its designed antagonist the area penalty (S5-N8). A 25% *firing* (halt)
  threshold would FALSE-FIRE on a working birth force. The alarm's own semantics ("the scored seg/pose signal
  is a passenger") are NOT triggered at 28% while seg is still 68-76% — seg is not a passenger. **Adopt
  instead: keep 40% FIRING, add a non-halting >25% WATCH band** (log the near-miss, never halt) so
  island_amplify's 28% is OBSERVABLE without false-halting. `verdict_scope: FORMULATION` (rejecting the
  lower-firing-threshold formulation; adopting the watch-band formulation).
- **Small-denominator guard (my refinement):** island_amplify is a NEGATIVE/reward term (pulls loss down),
  and terms sum > 1.0 in the backtest → the total can shrink near sign-cancellation, inflating `abs(term)/abs(total)`
  fractions spuriously. Require `abs(total)` above a floor before computing fractions, else the extended
  whitelist can false-fire on a near-cancelling total.
- **Fix the paraphrase everywhere:** keep "a single REGULARIZER term > ~40%" (exclude the primary objective);
  "any single loss term" is wrong.

### AMENDMENTS 1–3 — owed-1 (the pose-gate): see PART 3 (my sharpest attack drives these).

### AMENDMENT 6 — never-invent-flags footnote (one-final-cross-check outcome)

Every launch-1 ON-set flag is RE-VERIFIED against the live argparse (table in PART 4). **One footnote:**
`--muon-start-event` has `choices=["powerlaw_meat"]` ONLY (L11125). §B writes `"--muon-start-event": <nucleation>`
and A.5 STAGE 3 says "entry: tau nucleation / PLATEAU". **P7 must resolve `<nucleation>` to `powerlaw_meat`**
(the only valid choice; the help notes a "REV-B nucleation positive control" but the flag VALUE is
`powerlaw_meat`). Passing an invented `nucleation` string would fail argparse. `verdict_scope: INSTANCE`.

---

## PART 3 — THE SHARPEST NEW ATTACK (what the chain has NOT attacked): is the PRIMARY criterion measurable on the runs we WILL launch?

**The attack.** A-1's break was "the ABSOLUTE σ* floor (≥14.14) is unreachable → the gate never fires → the
pose apparatus is decorative → pose always ships banked R1." The v2 FIX promotes the **RELATIVE
relaxation-fit plateau** (`σ_min ≥ σ_min(∞)·(1−δ)`, σ_min(∞) = fitted asymptote of the run's OWN curve) to
PRIMARY. **But M1 proves this fix inherits the SAME never-fire failure**, one level down:

- The **only** σ_min trajectory in existence (the stopped run, 31 rows) is a **CV-0.21 NOISY OSCILLATION with
  a RISING drift**, NOT a relaxation curve. The exp-asymptote fit is DEGENERATE: `s_inf 0.210 ± 1.87` (891%),
  covariance non-estimable, beats a constant-model by only 18.9% (81% of variance is noise-around-a-constant).
- The fresh v7.5.2 run uses the **identical k_pairs=32 σ_min probe** → the same noise class. There is no
  reason to expect its σ_min to be cleaner, and **evidence (the one trajectory we have) that it is noisy**.
- ⇒ On the runs we actually launch, the exp-asymptote fit is likely DEGENERATE → the fit-quality guard (P4
  reformulation-queue item i) FALLS BACK to ship-banked-R1 → the PRIMARY criterion **never fires** → the pose
  apparatus is decorative AGAIN. **The A-1 fix RELOCATED the never-fire, it did not eliminate it.**

**The honest caveat on my own attack (attack-my-own-pass):** the one trajectory we have is from the **tau
stage** (`unify_tau`), not the **terminal band** where the gate actually fires (stage 4b, after Muon, on a
conditioned trunk). It is POSSIBLE the terminal-band σ_min is cleaner than the tau-stage σ_min. But (a) we
have zero terminal-band σ_min data, and (b) the burden is on the design to SHOW measurability, not assume it —
and the one data point we have says the k=32 sensor is too noisy for the as-specified fit. So the attack is
scoped as "**unvalidated + the only evidence says the sensor is too noisy for the exp-asymptote fit**," not
"guaranteed to fail." This is fixable by amendment, not a rewrite — and the ship-banked-R1 fallback (v2 Repair
2b) makes the launch SAFE even if the criterion never fires. `verdict_scope: FORMULATION` (the exp-asymptote
formulation of the plateau; a robust-plateau formulation is viable and is Amendment 1).

### AMENDMENT 1 — owed-1 PRIMARY criterion: robust rolling-slope on DE-NOISED σ_min (the load-bearing fix)

REPLACE the exp-asymptote-fit PRIMARY with a **rolling-mean-SLOPE ≈ 0 over a settle window** (P4 reformulation
queue i) computed on a **NOISE-REDUCED σ_min series** — EMA-smooth the σ_min series AND/OR raise the σ_min
probe's k_pairs to cut the CV-0.21 noise BEFORE any statistic is applied. A rolling-slope-≈0 detector is
robust to oscillation (it fires when the series stops TRENDING; it does NOT need to identify an asymptote
value) — and it correctly does NOT fire on the stopped run's RISING σ_min (conditioning still improving =
not-yet-plateaued, the right call). Add a **noise/fit-quality guard:** if the smoothed series' residual noise
exceeds a band (the detector cannot distinguish plateau from oscillation), FALL BACK to ship-banked-R1 (never
fire on a degenerate signal, never spuriously on flat noise). σ* stays advisory-only (v2 already correct).
`verdict_scope: FORMULATION`.

### AMENDMENT 2 — owed-1 canary: the achievable positive control (M2's "not $0" resolved honestly)

M2 is correct: **NO existing artifact can validate ANY plateau criterion** (R1 logged zero σ_min; the stopped
run won't fit; R1 kept only the ep1001 BEST ckpt so its σ_min TRAJECTORY cannot be reconstructed). The honest
re-spec of the "R1 positive-control canary":
- **$0 pre-launch controls (achievable now):** (a) a **NEGATIVE control** — assert the rolling-slope detector
  does NOT fire on the stopped run's 31 rising σ_min rows (existing data, $0); (b) a **SYNTHETIC POSITIVE** —
  feed the detector a synthetic clean relaxation curve, assert it fires. Together these are a real pre-launch
  unit test of the DETECTOR, satisfiable at $0.
- **The REAL positive control is GOVERNED, not $0, not pre-launch-blocking:** a telemetry-ON R1-equivalent
  replay (or resume from a preserved full-stage ladder if one exists) that LOGS σ_min(ep), then fit/plateau
  and assert it fires. This is owed but does NOT block launch-1.
- **The gap is covered by v2's own ship-banked-R1-PROCEED fallback (Repair 2b):** untrusted/absent canary ⇒
  ship banked R1 dxi + PROCEED + LOUD disengaged alarm. **State plainly:** launch-1 ships "banked-R1 unless
  the governed telemetry replay + the robust detector jointly validate an in-basin finish." The
  "pose-conditioned" headline is EARNED only after that governed validation; until then it is "banked-R1 with
  an OPTIONAL in-basin improvement," which is the honest floor. `verdict_scope: N/A` (feasibility).

### AMENDMENT 3 — owed-1 telemetry ON (the P5 "check the flag exists" mandate)

VERIFIED: `--jacobian-basin-telemetry` (L10213) default **True** — the σ_min sensor is ON by default; the
launch MUST NOT pass `--no-jacobian-basin-telemetry`. `--jacobian-basin-f-basin` (L10242) default 1.0 = the
running-MAX ratchet; the RELATIVE-plateau repurpose is the OWED BUILD (v2 correct). **DENSIFY the terminal-window
σ_min probe cadence:** the stopped run logged 31 rows over 124 ep (~1 row / 4 ep) — marginal for a rolling-slope
settle window; the terminal-band gate needs enough σ_min points inside its window for the slope statistic to be
meaningful. `verdict_scope: N/A` (config).

---

## PART 4 — FINAL never-invent-flags CROSS-CHECK (launch-1 ON-set vs live argparse)

All RE-VERIFIED against `experiments/train_levelset_witness_realized_through_R_mlx.py` argparse (line cites):

| flag | line | default | v2 sets | verdict |
|---|---|---|---|---|
| `--stability-preset` | 9998 | none (choices none,amber) | amber | ✓ |
| `--per-group-grad-clip` | 10056 | False | (amber-resolved) | ✓ |
| `--grad-normalize` | 10052 | none (per-param opt-in) | (not per-param) | ✓ (M3: amber does NOT set per-param — matches P4) |
| `--self-orient` | 10307 | False | true | ✓ |
| `--n-dir-freqs` | 10309 | 6 | 4 | ✓ |
| `--freq-across` | 10317 | 32.0 | 32 | ✓ |
| `--freq-along` | 10318 | **4.0** | **26** | ✓ (correctly OVERRIDES the along-tangent-starved default) |
| `--lane-render-band` | 11293 | False | true | ✓ (compile-coupled to self-orient) |
| `--dseg-aware-taper[-strength/-scale/-floor]` | 10328-10335 | False/1.0/0.0/0.05 | true/1.0/0.0/0.05 | ✓ |
| `--render-aa` | 11270 | none (choices none,supersample,ipe) | ipe | ✓ (supersample fail-closes w/ taper, L3084) |
| `--area-constraint-birth` / `-classes` | 11425/11431 | False / "1,3" | true / "1,3" | ✓ |
| `--birth-completion-event/-ramp/-classes` | 11450/11470/11467 | False/False/"1,3" | true/true | ✓ |
| `--persistence-classes` | 11378 | auto | "3" | ✓ |
| `--logit-adjust-classes` | 10929 | all | "3" | ✓ |
| `--seg-temporal-screw-weight` | 10706 | **0.0** | 0.1 | ✓ (help: cold-start 0.1; w=0 byte-identical) |
| `--seg-temporal-screw-start-event` | 10739 | None | annulus_plateau | ✓ (`annulus_plateau_event` is a real wiring, L5961/8626) |
| `--seg-temporal-screw-xi-source/-classes/-sky-rotation-only` | 10714/10723/10756 | ground_gt/"0,1,2"/store_true | ground_gt/"0,1,2"/true | ✓ |
| `--fused-r-kernel` | 10263 | False | true | ✓ |
| `--async-verdict` | 10255 | False | true | ✓ |
| `--safe-compile-manifest/-regions` | 10279/10273 | None/none | set | ✓ |
| `--micro-batch-pairs` | 10014 | 1 | EXCLUDED | ✓ (correctly excluded — #313 batch-dependence) |
| `--length-sigma-matrix` | 11036 | all-ones | (ladder 1b only) | ✓ (correctly OUT of launch-1) |
| `--jacobian-basin-telemetry` | 10213 | **True** | true | ✓ (sensor ON so the gate is measurable) |
| `--jacobian-basin-f-basin` | 10242 | 1.0 (running-max) | repurpose (OWED build) | ✓ (flagged as build) |
| `--pose-finish-start-epoch` | 9977 | 0 | backstop | ✓ (NO `--pose-finish-engage-on` — confirmed absent; engage is an OWED build) |
| `--muon-warm-start-momentum` | base 3187 | False | true | ✓ |
| `--muon-lr-final-frac` | 11144 | 1.0 | 0.1 | ✓ |
| `--muon-start-event` | 11125 | None (**choices=["powerlaw_meat"] ONLY**) | `<nucleation>` | ⚠ **AMENDMENT 6** — must resolve to `powerlaw_meat` |

**No invented flags. One value-footnote (Amendment 6).** The design is flag-clean.

---

## VERDICT

**SEAL-READY-WITH-AMENDMENTS (6, all amendment-grade — no rewrite).** The launch direction, the lever-class
split, the wall-clock design, the FRESH/F2-refuse leg, the speed neutrality receipts, and the ship-banked-R1
safety floor all survive fresh eyes. The three P4 FORCES-CHANGE are RE-DERIVE-CONFIRMED by me from primary
artifact and fold as Amendments 4/5/6. The load-bearing amendments are **1–2**: A-1's own fix (relative
exp-asymptote-fit PRIMARY) is, on the only evidence we have, **not reliably measurable on the runs we will
launch** — it must be re-specified to a robust rolling-slope on de-noised σ_min with a noise guard, and the R1
canary honestly re-scoped ($0 negative+synthetic controls pre-launch; governed telemetry replay as the real
positive control; ship-banked-R1 covers the gap). None of the 6 invalidates the launch; all are correctable
in P6. The launch is honest that it ships **banked-R1 unless a governed replay + a robust detector jointly
earn the in-basin "pose-conditioned" finish** — never pose-blind-with-a-decorative-gate.

**Pointer 0.19110 UNMOVED — this red-team is MEANS. Only a byte-closed `upstream/evaluate.py` n600 row
< 0.19110 moves it.** Every amendment above is unreviewed new work; P6 must RE-DERIVE from the primary
artifacts, not trust this memo.

## STORES CONSULTED (line)
SYNTHESIS_v2_v752 (90d562a5a) · P4_recess (4a089be2b) · P3_redteam_verdict · ORCHESTRATION_LEDGER
(§178/188/205 operator bindings) · PRIMARY CODE re-derived {mlx/optimizers/optimizers.py L896-948 ·
train_witness…mlx.py L1879-1887/L2414-2418 · train_levelset…mlx.py argparse+L9222-9236+L8012-8013+L5065} ·
levelset_pose_gate_l7best_20260701.partial.jsonl (via P3/P4) · delta_R_noise_floor.json (via P3). $0, no GPU,
no training, run dirs READ-ONLY, #205 UNTOUCHED, [no-triality].
