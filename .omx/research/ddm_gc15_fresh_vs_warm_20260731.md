---
council_tier: T3
council_attendees: [Schmidhuber, Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Quantizr, Hotz, MacKay, Ballé, Boyd, Tishby-memorial, Time-Traveler, PR95Author]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: true
council_override_rationale: "Does the pantheon think we may need to try birthing from a fresh run start to get truly optimal or is continuing to train against and loop against warm start optimal" + (extension) "Should we be resetting every time, or should we reset against upstream weights and gradients for dynamic correction and maybe wait certain channels and such based on upstream channels and hyperplanes and weights and basis and gradients?"
model_provenance: "Opus 5 (claude-opus-5). Fable-5 is at its usage limit; gc14 also ran on Opus. Model-identity honesty per the ddm_b4s death record."
related_deliberation_ids: [ddm_gc14_first_descent_20260731, ddm_fl1_perclass_flicker_floors_20260731, ddm_b4s_burn4_charter_20260731, ddm_bc1_qa24_compose_and_fire_20260731, ddm_j4_366_warm_start_reform_20260723]
---

# ddm_gc15 (#816) — fresh vs warm: the dichotomy is false, and the reset operator is an unpriced 6.57× LR spike

**17th operator-convened pantheon convocation. Scorer-FREE (0 SegNet/PoseNet forwards — window_03 owns the slot). $0. Pointer `0.1910828242` [contest-CPU] UNMOVED — this convocation is MEANS.**

---

## §0 HEADLINE (answer first)

The operator asked two questions on the same object. The answer to both is the same object, and it is neither pole.

**1. The dichotomy is false, and MAIN's seed is right about that — but for a stronger reason than the seed gives.** The seed says we ran a hybrid (warm weights × repeatedly-fresh optimizer). True. What the seed did not know is **what the fresh optimizer actually does**, and it is not a neutral "restart":

> `experiments/train_tr1_partition_renderer_mlx.py:1543` builds `optim.Adam(learning_rate=cfg.lr)`.
> **MLX's `Adam` defaults `bias_correction=False`** (verified from `mlx.optimizers.Adam` source), and the trainer never overrides it. With β=(0.9, 0.999) and zeroed moments, the effective step-size multiplier is
> **η(t) = (1−β₁ᵗ)/√(1−β₂ᵗ)** — **η(1) = 3.162, peaking at η(12) = 6.569**, decaying to 1 with time-constant 1/(1−β₂) = 1000 steps.

At the burn's measured geometry (`--batch-pairs 8`, `--num-pairs 600` ⇒ **75 steps/epoch**; `--lr 2e-3`; 140-epoch window = 10,500 steps), **each window boundary injects 1,212.6 extra sign-steps of parameter displacement = 16.17 epochs of free movement = an 11.5% displacement bonus per window, of which 81.7% lands in the first 13 epochs.**

**This derives gc14's entire measured shape with no new experiment**: a large step at the boundary; flat within the window (η ≈ 1 by epoch ~67, and the residual 18% is smeared over 127 epochs); magnitude set by restart count not epoch count (the integral converges to 1,212.6 regardless of window length); |step|/|window Δ| > 1. gc14 called the mechanism INFERRED and owed R1 to establish causation. **The causation is now DERIVED from source + closed-form, and R1's job changes from "is it the restart?" to "which reset is optimal?"**

**The uncomfortable corollary, stated plainly:** the campaign's only measured seg descent this month is, on the leading derived hypothesis, **an artifact of a missing bias correction** — a two-character default in a third-party optimizer that no decision record has ever named. A correctly bias-corrected resume would have produced η ≡ 1 and, by this mechanism, **no boundary step at all**.

**2. The reset is an OPERATOR with three independent knobs, and we have been running its most generic instance in violation of our own standing law.** Knob 2 (what we reset *to*) is currently **zero**. Because `bias_correction=False`, a `v=0` reset makes the first step `lr·(0.1g)/(0.0316|g|) = 3.16·lr·sign(g)` — **a uniform-magnitude, sign-only, metric-free step: the maximally generic step there is.** Our standing law `generic_basis_metric_never_optimal` (operator 07-29: "cosine+Fourier+Euclid — derived-or-raced binding") and sy1's `S1-POLICY` ("Any Euclidean-default projection is REFUSED, verdict_scope=G5 engine metric declaration") both forbid exactly this. **Nobody ever derived or raced the zero-reset. It arrived as a library default.**

**3. Therefore: a STRUCTURED reset plausibly dominates BOTH poles, and that is the interesting outcome the operator flagged.** Warm-with-k-restarts gets k kicks and keeps the path; fresh gets 1 kick and must re-learn the path. So warm already dominates fresh *on the restart axis* — and the restart axis is a KNOB, not a property of warmness. A fresh run with the same cadence gets the same bonus. What is left for "fresh" to buy is only what warm structurally cannot buy (§10). Meanwhile the direction of the kick is completely unaimed, and re-aiming it costs **zero additional wall-clock**, because the reset happens anyway. **The cheapest unexploited lever in the campaign is not more epochs and not a fresh birth — it is the metric of a reset we are already paying for.**

**4. The QA24 from-birth "MEASURED-DOMINATED" verdict does not survive matched-compute normalization, and the sign flips (§8).** bc1 fresh was measured at **ep399**; the warm base at **ep641** — the warm arm had **60.7% more training**. At the nearest matched-compute point the warm lineage has a receipt (ep499, seg 0.49410), and projecting fresh forward at the warm lineage's *own, conservative, later-stage* rate puts **fresh ahead by 0.0249 S on seg and 0.0127 S on rate — ~0.035 S ahead in total**, reversing a 0.078 S "domination." **DERIVED-BY-EXTRAPOLATION, INSTANCE scope** — but it is enough to move QA24-from-birth from CLOSED to **CONDITIONALLY RE-OPENED**.

**5. The one cell where fresh is not a preference but a structural necessity is QA84 rowband** — `pa1r`: "NO D8/rowband checkpoint exists anywhere (every tr1 ckpt is D16) ⇒ the rowband arm is a fresh full-curriculum decision, not a warm tail." That is the fresh-run decision function's primary trigger (§10).

---

## §1 PROVENANCE AND AUTHORITY

| item | value |
|---|---|
| venv | `/Users/adpena/Projects/pact/src/tac/__init__.py` — hijack check **CLEAN** |
| git HEAD at start | `e922da7a92` |
| scorer jobs run | **0** (window_03 owns the single n600 slot) |
| burn custody | `/Volumes/VertigoDataTier/pact/ddm_b4s_20260731/` — **READ-ONLY**, zero writes |
| evidence axis | `[macOS-CPU/MLX advisory]` + `[source-derived]`; `score_claim=false`, `promotable=false` |
| pointer | **`0.1910828242` [contest-CPU] UNMOVED** |
| bar | `min(0.15, official 0.172141)`; own-vehicle exact-protocol line **0.9639878** |

**Primary sources used for the load-bearing derivation (all read this session, not recalled):**
`experiments/train_tr1_partition_renderer_mlx.py` L1536–1585 (opt_state_flat={} / fresh Adam / documented "warm-start re-anchor law #517/#518") · L1213–1214 (`--lr` default 2e-3, `--batch-pairs` default 8) · `mlx.optimizers.Adam` source via `inspect.getsource` (`bias_correction: bool = False`, `init_single` zeros m and v) · `/Volumes/VertigoDataTier/pact/ddm_b4s_20260731/tickets/window_01_ticket.json` (batch-pairs 8, num-pairs 600, lr 0.002, grid-downsample 16, epochs 666).

**Flag census (never-invent-flags):** tr1 exposes **64** argparse flags. A grep for `adam|beta|bias|moment|restart|precond|warmup` over them returns **ZERO matches**. The entire reset operator is unexposed, unconfigurable, and undeclared. **Every lever proposed below is a to-be-BUILT DSL `Lever` factory, never a hand-added trainer flag** (config-orphan law).

---

## §2 PRIOR-LAW PREDICTION LINES (stated BEFORE composing; diffed after)

| law | what it predicted about fresh-vs-warm | diff |
|---|---|---|
| **never-launch-weaker-state** | Fresh is a weaker state by construction; pursue optimal from the start ⇒ prefer warm | **CONFIRMED but RE-SCOPED.** The law is about *state*, and it is right. It does not license the *reset* being generic. The correct reading: a fresh birth is a weaker state; a badly-aimed reset on a strong state is *also* a weaker action. The law now binds knob 2, which nobody had applied it to. |
| **constants-are-poison** | Thresholds/constants must be DERIVED | **NEW, LARGEST INSTANCE FOUND.** β=(0.9,0.999) and `bias_correction=False` are **inherited third-party library defaults** governing a 6.57× effective-LR excursion. This is the biggest un-derived constant the campaign has been running. Sister to gc14's `n_points=5` and window-length findings; strictly larger than both. |
| **generic-basis-metric-never-optimal (07-29)** | A generic metric is forbidden absent derivation or race | **VIOLATED, undetected until now.** `v ← 0` + no bias correction = a *sign* step = the generic metric. Never derived, never raced. |
| **verdict-scope ladder** | One failed formulation ≠ dead family | **DECISIVE for two re-opens.** (i) K-FAC "DISCARD stands" (negative_audit_wave 07-13) was scoped to a *per-step preconditioner with no ticket consumer*; a *boundary-insertion* is a different formulation in the same family and now has a consumer. (ii) QA24-from-birth "MEASURED-DOMINATED" was INSTANCE-scoped at unmatched compute. |
| **ERF-collateral** | Recovery must be BORN in-loop; post-hoc injection is net-negative | **BINDS THE FRESH ARGUMENT AND LIMITS IT.** It is the strongest support for "protect from birth" (seed item 2) — and simultaneously the reason a *reset* is legitimate where an *injection* is not: a reset changes the in-loop optimizer, it does not paint. |
| **non-additive pools** | Same-pool levers compete, never sum | **BINDS §9.** Restart-cadence, reset-metric, and per-coordinate structure are all in the SAME pool (they all act on the same boundary event). Their gains must be RACED, never summed. |
| **staleness-at-consumption** | Consume fresh, fail closed | **NEW INSTANCE.** Adam's `v` is a *stale empirical Fisher* at a boundary — that is the honest argument FOR discarding it. But "stale" ⇒ *refresh*, not ⇒ *zero*. The current code takes the wrong branch of our own law. |
| **alarm-predicates-are-per-vehicle-calibration-objects** | First fire = calibration event | **EXTENDS.** Generalizes from alarm predicates to **optimizer hyperparameters**: a third-party default is a per-vehicle calibration object too. |
| **no-old-lineage ban** | HNeRV/PR95/110/128 = lessons-only | **HONORED.** Nothing below transplants a PR-lineage schedule. SGDR/SWA/K-FAC/lottery-ticket are consumed as *mechanism questions*, never as configs. |
| **gc13 Pontryagin TPBVP** | Forward primal + backward dual settle at window boundaries | **AMENDED FURTHER than gc14 amended it.** gc14 found the boundary is a state discontinuity. gc15 makes it quantitative: the jump is a **known, closed-form, 1,212.6-sign-step impulse**. The TPBVP's impulse term is now *writable*, not merely *owed*. |
| **gc14 boundary-step finding** | The level moves at boundaries; mechanism INFERRED; R1 owed for causation | **UPGRADED INFERRED → DERIVED.** And gc14's r = 0.310 is re-read: the per-restart *impulse* is CONSTANT (1,212.6 steps); r measures the **landscape's** diminishing return to a fixed displacement, not a decaying kick. That is a materially different and more useful reading. |

**Anti-re-anchor check (the 07-30 law):** did I re-find an existing law? **Partly, and I say so.** The trainer's own comment at L1583 already says *"Adam moments are re-anchored fresh (warm-start re-anchor law #517/#518)"* — the reset was a **deliberate, documented, named decision**, not an accident. What was never done is **pricing it**. gc15's contribution is not discovering the reset; it is (a) discovering that `bias_correction=False` turns a documented re-anchor into an undocumented 6.57× LR spike, and (b) pricing it at 16.17 epochs/boundary. Law #517/#518 predicted the *existence*; it did not predict the *magnitude*, and the magnitude is what changes the decision.

---

## §3 MULTI-PASS RECALL (what each pass returned)

**Pass 1 — the mandated deferrals + actuators.**
`ddm_b4s_burn4_charter` §1/§4 (QA24 from-birth **MEASURED-DOMINATED** bc1 0.686 > warm 0.608; **from-birth-KD DEFERRED** "requires from-SCRATCH, a separate charter"; **rowband DEFERRED** "from-scratch; not a continuation cell"; §3b R7 KD wired but continuation-KD dw1-CLOSED) · `ddm_bc1_qa24_compose_and_fire` + DAG_FEED (400ep/480min, solve_project init, ep0 loss 60.3) · **`kd_warm_start_actuator_20260616T210540Z.md`** — the #74/#129 KD-warm-start actuator: `cfg.kd_warm_start_dir`, latents load directly (taper-independent), decoder distilled from a frozen teacher, **BUILT + 6 NO-FAKE tests, default-OFF, never fired on tr1** · `ddm_deferral_queue_ledger_20260729.md` QA84 (rowband **BUILT, burn-2-ready**, `RowBandGrammar`, `--token-rowband-spec`, DOF 1248; **from-birth arm BLOCKED-measured — no D8 parent exists**) and QA89/pa1r (rowband **BLOCKED — fresh-burn class**).

**Pass 2 — laws + the second axis.**
`ddm_j4_366_warm_start_DAG_FEED` (an actual warm-start *reform*: explicit β₂, **β₂-derived 2000-step linear LR ramp**, quarter-quantum cap, template freeze, delayed pose, "**Restore complete Adam/EMA/cursor state or initialize fresh moments**" as an explicit branch; smoke `BLOCKED_REALIZED_NO_COMPONENT_DESCENT`; reuses canonical equation **`adam_v_variance_warmup_length_v1`**) · `ddm_hb1_hope_bn_capacity_findings` (#725 **BUILT**: `src/tac/optimization/hope_bn_capacity.py`, 78 BN units, exact n600 empirical kernels, argmax custody 0.999999991522895, rank-4 head singular values 4.703/2.831/2.039/2.018, per-stratum `cap_b^{ab}(i)=‖Δw_ab[i]‖_F·√(K_b(i))`, **0 dead channels**, capacity strongly non-uniform: Lane–Movable top-3 channels = 70.7%) · `ddm_ms3` (PF2 exact 1,200-bucket atlas; bundle PARTIAL) → **`ms4d` BUNDLE-COMPLETE** (the metric bundle is COMPLETE and passes the strict MS3 loader) · `ddm_lg1_lane_guard` item 4 (gradient surgery **DEFERRED**: 2-backward Fisher projection, **~1.8× step wall-clock**, reuse `contain_protected_grad_mx` from `tac.boundary_math.island_protection` L594 — do NOT fork) · `sc2` §14 (`token_init_mode=solve_project` **ADOPTED at −28.9%** n600 at matched epoch — the campaign's one measured *init* lever) · `negative_audit_wave_20260713` row 12 (**K-FAC DISCARD**) · `sy1` S1-POLICY (Euclidean-default projection REFUSED) · `oss_untried_technique_candidates` §5c (**SGDR `1608.03983` logged, never fired**; antagonism flag: "a restart can *un-place* an already-correct coarse partition if fired too late") · `curriculum_openpilot_seeded_deepmath_dsl` (**a FULL restart (1.0×) reproduced the v3 destabilization** ⇒ partial 0.1× floor adopted — a measured receipt AGAINST naive full restarts).

**Pass 3 — targeted (this session's derivation).** MLX `Adam` source; tr1 L1536–1585 + L1213–1214; the window_01 ticket argv; the 64-flag census.

---

## §4 AUDIT OF MAIN's SEED (typed)

| # | seed claim | verdict | receipt |
|---|---|---|---|
| **1** | "The dichotomy is false — we ran warm weights × repeatedly-fresh optimizer; the live variable is RESET CADENCE" | **DERIVED-with-receipt, and STRENGTHENED** | Correct and now mechanistically closed. Correction to the seed: cadence is only **one of three** knobs, and it is **not the highest-information one** (§5.5). The seed's continuum is a line; the object is a 3-cube. |
| **2** | "The fresh asymmetry — the full protection/force stack has never been exercised TOGETHER FROM ep0; protect-from-birth ≠ protect-after-erasure. Am I overrating it?" | **PLAUSIBLE-with-named-$0-measurement — and yes, mildly overrated, for one specific reason** | The *premise* is DERIVED: §3b of the b4s charter shows R1/R2/R3/R8/R14 are all DESIGNED-STUB (not trainer-wired), R6 is the only engaged seg lever, and gc14 proved **λ_Lane was 0.0 at all 38 gates** — so the stack has never run together *at all*, from ep0 or otherwise. **The overrating is this:** "from ep0" is doing less work than the seed thinks, because *most of the stack is not built*. The binding blocker is **wiring, not birth**. Building the stack and firing it warm is strictly cheaper and tests the same forces. Where "from ep0" genuinely binds is narrower and real: **erasure is path-dependent and the ERF-collateral law says recovery must be born in-loop** — but gc14's own strongest receipt (`gt_components_erased` 567→508, Lane −53, **in-loop, warm, unaided**) is direct evidence that a *warm* run recovers erased structure. That receipt argues against the necessity of birth. **Named $0 measurement:** the erasure ceiling — from the ru1 atlas, what fraction of erased GT components are in cells the *current warm* trunk has already re-birthed at least once? If high, birth buys little. |
| **3** | "The warm counter is scoped — QA24 was measured on a D16-capped grid; re-grade" | **DERIVED, and the re-grade is far stronger than the seed's framing** | The D16 cap is real (`--grid-downsample` choices={8,16}) but it is **not the binding confound**. The binding confound is **compute: ep399 vs ep641 (+60.7%)**. Under matched-compute normalization the verdict's **sign flips** (§8). |
| **4** | "The axis may dominate both — if B5-C hands the slot to RATE, does fresh-vs-warm on SEG even matter?" | **DERIVED — and this is the seed's best instinct** | Yes, and it resolves cleanly: fresh-vs-warm **on the seg axis** is a ~0.01–0.02 S question (gc14's geometric remaining 0.00946 S) against a **0.098 S** banked rate pool. **But the operator's rowband cell is a RATE cell that is structurally fresh-only** — so "hand the slot to RATE" and "do we need a fresh birth" are not competing answers; the rate handoff is *itself* where the fresh question becomes live. That is the reconciliation. |

---

## §5 THE RESET OPERATOR — formalized on three knobs

Define the boundary event as an operator **R** applied to the training state `(θ, m, v, θ̄_EMA, τ_EMA)`:

```
R(θ, m, v, θ̄, τ) = (θ', m', v', θ̄', τ')
  knob 1  WHAT RESETS      : subset of {m, v, θ̄, θ}         [currently: {m, v} zeroed; θ̄ KEPT; θ KEPT]
  knob 2  WHAT IT RESETS TO: zero | previous | scorer-prior  [currently: ZERO, undeclared]
  knob 3  PER-COORDINATE   : uniform | per-channel weighted | staggered release
                                                             [currently: UNIFORM, undeclared]
```

All three currently move together, unpriced and unseparated. **Separating them is itself a measurement nobody has taken.**

### §5.1 The derived mechanism (the headline; no experiment required)

With `bias_correction=False` (MLX default, not overridden) and `init_single` zeroing both moments, the post-reset update is

```
Δθ_t = −lr · m_t /(√v_t + ε),   m_t = (1−β₁ᵗ)·g,  v_t = (1−β₂ᵗ)·g²   (constant-g idealization)
     = −lr · η(t) · sign(g),    η(t) = (1−β₁ᵗ)/√(1−β₂ᵗ)
```

| quantity | value (β=(0.9, 0.999), 75 steps/epoch, lr 2e-3) |
|---|---:|
| η(1) — first step after reset | **3.1623×** |
| η peak (t = 12 ≈ 0.16 epoch) | **6.5685×** |
| mean η over the first epoch | 5.058× |
| mean η over the first 5 epochs | 2.908× |
| mean η over the first 13 epochs | 2.016× |
| η → 1 (within 0.4%) | t ≈ 5,000 steps ≈ **67 epochs** |
| **extra displacement per boundary** | **1,212.6 sign-steps = 16.17 epochs = 11.5% of a 140-ep window** |
| **fraction delivered in first 13 epochs** | **81.7%** |

**Four independent gc14 observations that this closed form predicts, none of which it was fitted to:**
1. Step at the boundary, flat inside (81.7% of the impulse in the first 13 of 140 epochs).
2. `|step| / |window Δ|` > 1 (gc14 measured 1.36).
3. Loss spikes at resume (+9.6%, +22.8%) **at normal gnorm ratios 0.94–0.99** — gc14 correctly called this "optimizer step-scaling, not data." η is exactly that step-scaling, and it is 3.16–6.57×.
4. "Magnitude set by restart count not epoch count" — the impulse integral converges (1,212.6 at 140 ep vs 1,209.3 at 67 ep: **99.7% of the impulse is delivered by epoch 67 regardless of window length**).

**Bias correction removes it exactly.** With `bias_correction=True`, at t=1: m̂ = m/(1−β₁) = g, v̂ = v/(1−β₂) = g², step = lr·sign(g) ⇒ **η ≡ 1 for all t**. This is the cheapest possible falsifier of the whole mechanism and it is a one-field change.

**Honest scope:** the constant-g idealization is exact only for slowly-varying gradients. Real g fluctuates, which *lowers* the m/√v ratio (a variance penalty) but does not remove the (1−β₂ᵗ) denominator bias, which is the dominant term. The η profile is therefore an **upper envelope**; the true multiplier is between 1 and η(t). This does not change any sign or any ordering below, and the A/B in §7 measures the realized value directly. `verdict_scope: STRUCTURAL` (the code path is universal to this trainer), `empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION` for the mechanism, `ASSUMED_AWAITING_VERIFICATION` for the realized magnitude.

### §5.2 Knob 1 — WHAT RESETS

| component | current | what it means | separable? |
|---|---|---|---|
| **m** (momentum) | zeroed | discards direction memory | yes — cheap |
| **v** (second moment) | zeroed | **discards the diagonal empirical Fisher** — this is the one that creates η | yes — cheap, and it is the load-bearing one |
| **θ̄ (EMA shadow)** | **KEPT** (`ema = st["ema"]`) | the gate reads `gate_params: "ema_shadow"` | yes |
| **τ_EMA (decay)** | **RE-DERIVED per window** (166→202→236 ep, gc14 V4) | lengthening average = unintended SWA | yes |
| **θ (live weights)** | KEPT | this is what "warm" means | yes (= the fresh pole) |

**The finding:** m and v are zeroed *together*, but they do different things. Zeroing **m** is a genuine direction-memory reset (defensible under staleness). Zeroing **v** is what produces the 6.57× spike — and `v` is not "stale direction," it is **a scale estimate**, whose staleness across a 140-epoch window is far milder than momentum's. **The current code applies the harsher treatment to the quantity that needed it less.** That asymmetry has never been stated.

### §5.3 Knob 2 — WHAT IT RESETS TO (the operator's proposal, derived from OUR quantities)

Adam's `v` **is** a diagonal preconditioner ≈ diag of the empirical Fisher. We have the actual frozen-scorer metric in custody (`ms4d` BUNDLE-COMPLETE, passing the strict MS3 loader). So "reset against upstream" = **inserting a natural-gradient/mirror-descent metric at the one point where the empirical metric is being thrown away anyway.** It is the cheapest possible insertion point in the entire training loop — strictly cheaper than lg1's deferred per-step gradient surgery (**~1.8× step wall-clock**), because a boundary insertion is **O(1) per window, ~0× wall-clock**.

Three candidate targets, ordered by cost:

- **(2a) `v ← v_prev` (carry-forward).** Persist `v`, still zero `m`. **$0 of new math** — a persistence change only. Sets η ≡ 1 by construction (v is already at scale) while keeping a true momentum reset. **This is the control that isolates "is the gain the spike?" from "is the gain the momentum reset?"**
- **(2b) `v ← v_scorer` (scorer-derived prior) — DERIVED, ours, not transplanted.** From #725, for each pre-head channel *i* and class-pair stratum *b*, the measured per-stratum capacity is `cap_b^{ab}(i) = ‖Δw_ab[i]‖_F · √(K_b(i))`, where `K(i,i)=E[ψ_i²]` is an **exact empirical second moment under the n600 measure** (argmax custody 0.999999991522895 — the integration measure *is* the frozen scorer's own measure) and `Δw_ab` comes from the **exact rank-4 head** (measured singular values 4.703/2.831/2.039/2.018). `cap²` has the units of a diagonal Fisher entry. Setting `v ← s·cap²` (s a single scalar fixed by norm-matching, per §7's magnitude-matching rule) makes the post-reset step **isotropic in the scorer's output metric rather than in parameter space** — natural gradient, evaluated at the boundary, from quantities we already own.
- **(2c) `θ ← θ_prior` (weight rewinding).** The lottery-ticket instance of knob 2. **Ruled out for now** by the measured `curriculum_openpilot_seeded_deepmath_dsl` receipt: *"a FULL restart (1.0×) reproduced the v3 destabilization"* ⇒ partial (0.1×) adopted. Weight-level resets on this campaign have a measured destabilization receipt. Keep as a named-but-not-fired cell.

**The honest tension I will not paper over.** Two readings of the mechanism give **opposite** per-coordinate prescriptions:
- *Conditioning reading* (natural gradient): equalize the effect on the scorer's output ⇒ **small** parameter steps in high-capacity channels ⇒ `v ∝ cap²`.
- *Targeting reading* (spend the kick where the scorer can see): **large** steps in high-capacity channels ⇒ `v ∝ 1/cap²`.

They agree on exactly one thing, and it is the robust core: **coordinates whose image lies in the scorer-null subspace should receive ZERO kick** — the conditioning reading because a flat direction has no meaningful natural step, the targeting reading because it is pure waste. The *shape within* the visible subspace is genuinely contested and must be **raced, not presumed** (generic-basis law). §7 races both signs.

**Precision on the null subspace (do not conflate — I nearly did).** `ker(A) ≈ 52%` and `80.67% resize-null per channel` are properties of the **realization/resize operator R on the rendered image**, i.e. which parts of the *image* the scorer cannot see. `hb1` separately measured **0 dead channels** inside SegNet (no analytically dead capacity, K ≥ 1e-12 everywhere). So the waste argument runs through the **composite R∘render adjoint**, not through SegNet channel death. That composite adjoint is exactly the `ms3` row that `ms4d` completed. **Correctly stated: a large fraction of a uniform parameter-space kick lands in image directions the scorer cannot see — and that fraction is measurable at $0 from the completed metric bundle, before any arm runs.**

### §5.4 Knob 3 — PER-COORDINATE STRUCTURE (both readings of "wait certain channels")

The operator's phrase admits two productive readings; **both are treated, neither is presumed.**

- **(3a) WEIGHT — per-channel preconditioner scaling** by scorer sensitivity. The `v ← s·cap^±²` construction above. #725 measures capacity as **strongly non-uniform** (Lane–Movable stratum: top-3 of 16 channels carry **70.7%**; Lane–Undrivable: channel 9 alone **30.0%**; Road–MyCar much flatter, top channel 16.1%). A uniform kick is therefore demonstrably mis-allocated relative to the measured structure — and the per-class heterogeneity means the *right* weighting is class-dependent, which composes with the fl1 per-class debt ranking (Lane #1).
- **(3b) WAIT — staggered release in scorer-sensitivity order.** Freeze all coordinates at the boundary, then release them in descending `cap` order over the first N steps, so the amplified early steps are spent only on the coordinates the scorer weights most. This composes directly with **j4's measured freeze-then-release** ("activate island coordinates only; freeze shared-template coordinates") — noting j4's smoke returned `BLOCKED_REALIZED_NO_COMPONENT_DESCENT` at **INSTANCE** scope on a *different vehicle* (v10 receiver), so it is a caution, not a closure, for tr1. It also composes with j4's **β₂-derived 2000-step linear LR ramp**, which is an *independent prior derivation of the same 1/(1−β₂) = 1000-step timescale I derived here* — two routes to the same number is the strongest corroboration in this memo.
- **Relation to lg1's deferred gradient surgery:** lg1 item 4 (2-backward Fisher projection, ~1.8× step cost) is the per-step version of the same idea. **A boundary insertion achieves a large part of the aim at ~0× wall-clock.** If the boundary version pays, lg1's surgery becomes a *refinement* with a measured prior rather than a speculative 1.8× tax.

### §5.5 KNOB RANKING by expected information per $0 (mandated)

| rank | knob | why | cost |
|---|---|---|---|
| **1** | **Knob 2 ×  bias-correction (the `v` treatment)** | It is the *cause* of the whole phenomenon. It has a **closed-form prediction** (η) that a single run either confirms or refutes. It is a one-field change. And it is the standing-law violation. **Highest information per dollar in the entire campaign right now.** | $0 build, ~2h/arm |
| **2** | **Knob 3a per-channel structure** | #725 is **already built and measured**; the non-uniformity is large (70.7% in 3 of 16 channels). But the *sign* is contested (§5.3), so one arm cannot settle it — it needs two. | $0 build (#725 exists), 2 arms |
| **3** | **Knob 1 (m vs v separation)** | Cheap and clean, but it is largely *subsumed* by rank 1: the `v ← v_prev` arm already separates them. | free rider on rank 1 |
| **4** | **Knob 0 — CADENCE (MAIN's seed / gc14's R1)** | The impulse per boundary is now **known and constant** (1,212.6 steps). Cadence therefore has a *predicted* answer (k restarts ⇒ k impulses, sublinear in S via the landscape's r), so an A/B on cadence alone mostly re-measures a derived quantity. **It should still run, but as the control arm of the metric experiment, not as the experiment.** | 2 arms |
| **5** | **Knob 3b staggered release** | Most design surface, least prior; j4's INSTANCE negative counsels sequencing it behind ranks 1–2. | needs build |

**This ranking is the single most actionable output of this convocation, and it demotes MAIN's own seed item 1 from first to fourth — with a reason, not a preference.**

---

## §6 THE RESET-CADENCE CONTINUUM, formalized

Let the campaign be parameterized by **(ρ_opt, ρ_θ, M)** — optimizer-state reset rate, weight reset rate, and the reset metric.

```
ρ_opt = resets per 1000 epochs (optimizer state)
ρ_θ   = resets per 1000 epochs (weights to init)      ρ_θ ≤ ρ_opt always
M     = the metric of the reset ∈ {zero, prev, scorer-prior}
```

| regime | ρ_opt | ρ_θ | M | status |
|---|---:|---:|---|---|
| pure warm, textbook | 0 | 0 | — | **never run** (would require persisting Adam state; the code cannot do it) |
| **what we actually ran** | **~7.1** (1 per 140 ep) | 0 | **zero** | burn-4 windows 01–03 |
| bc1 from-birth | ~2.5 (1 per 400 ep) | 2.5 | zero | fired once, ep399 |
| R1 half-window arm | ~14.2 | 0 | zero | **#815, queued** |
| SGDR-like | 10–50 | 0 | (LR-schedule, not moments) | never fired here |
| **fresh from birth** | **≥ρ** | **= 1 per run** | zero | the operator's pole |
| **structured warm (proposed)** | 7.1 | 0 | **scorer-prior** | **§7 arm D — never run** |

**Theory's prediction for the optimum.** Total impulse ≈ `ρ_opt × I` where `I = ∫(η−1)dt = 1,212.6` is **fixed and cadence-independent** (99.7% delivered within 67 epochs). Marginal S-return per impulse decays at the measured landscape rate `r = 0.310` per gc14. So `S(ρ_opt)` is **increasing but strongly concave**, and is bounded above: gc14's geometric envelope caps 2× cadence at `2 × 0.00946 = 0.019 S`. Meanwhile ρ_θ > 0 (fresh) **subtracts** the path term. **Prediction: the optimum is interior in ρ_opt, at ρ_θ = 0, and — crucially — the whole curve shifts when M changes. M is a bigger lever than ρ because it changes the *efficiency* of every impulse, not the count.**

**Where R1's result will place us.** Under the derived mechanism, R1's half-window arm gets **2× the impulses = 2,425 extra sign-steps**. Pre-registered: it should beat the unbroken control, but by **less than 2×** the single-window gain (concavity), landing in **[1.3×, 1.9×]**. Three outcomes and their readings:
- **beats control by <1.3×** ⇒ landscape saturation dominates; cadence is nearly exhausted; go to M.
- **beats by 1.3–1.9×** ⇒ mechanism confirmed as derived; cadence is a real but bounded lever; M is the growth axis.
- **beats by >2× or fails to beat at all** ⇒ **η is not the mechanism**; §5 is falsified and gc14's R1-as-originally-designed is back to being the right experiment.

---

## §7 THE A/B DESIGN — four arms, magnitude-matched, pre-registered

**The operator's mandatory falsifier is upgraded from "possible" to "the leading derived hypothesis."** §5.1 says the benefit *is* an LR spike — i.e. it *is* disorder. A naive scorer-metric arm would damp exactly that and lose for an uninformative reason.

**The design fix (this is the load-bearing methodological move):** **hold the total kick norm ‖Δθ‖ fixed across all metric arms and vary ONLY the direction distribution.** Concretely, scale each arm's `v` by a single scalar `s` chosen so the first-100-step cumulative ‖Δθ‖ matches the zero-reset arm to within 2%. Then the experiment is a clean test of **where the kick goes**, not **how big it is** — and the falsifier is defused by construction rather than by argument.

| arm | knob 1 | knob 2 | knob 3 | tests |
|---|---|---|---|---|
| **A — no-reset control** | persist m and v | previous | uniform | Is the boundary step real at all? (requires the P0-compliant opt-state persistence the code lacks) |
| **B — zero-reset (current default)** | zero m, v | zero | uniform | The accidental incumbent |
| **B′ — bias-corrected zero-reset** | zero m, v | zero | uniform | **η ≡ 1 with everything else identical — the surgical isolator of the spike** |
| **C — momentum-only reset** | zero m, **persist v** | previous | uniform | Separates knob 1's two components |
| **D+ / D− — scorer-metric reset** | zero m, `v ← s·cap^{+2}` / `s·cap^{−2}` | **scorer-prior** | per-channel | Conditioning vs targeting, at matched ‖Δθ‖ |

**Pre-registered predictions (stated before any arm runs).**

| hypothesis | A | B | B′ | C | D+ | D− |
|---|---|---|---|---|---|---|
| **H1 — the spike is the mechanism** (my leading hypothesis, P ≈ 0.6) | worst | good | **≈ A (no step)** | ≈ A | ≈ B | **best** (kick aimed at visible directions) |
| **H2 — conditioning is the mechanism** (P ≈ 0.2) | worst | good | good | **good** | **best** | worst |
| **H3 — momentum staleness is the mechanism** (P ≈ 0.1) | worst | good | good | **best** | ≈ C | ≈ C |
| **H4 — none of it; the step is something else** (P ≈ 0.1) | ≈ B | ≈ B | ≈ B | ≈ B | ≈ B | ≈ B |

**B′ is the highest-information single arm** and it is nearly free: it distinguishes H1 from H2/H3 by itself, with a one-field change and no new math. **If B′ ≈ A (the step vanishes), the campaign's descent is confirmed as a bias-correction artifact — and that is the most decision-relevant single fact available at $0 anywhere in the campaign right now.**

**Derived preflight (compute BEFORE any arm — the gc14 §15 discipline, extended).** Two $0 numbers, both from banked artifacts:
1. **Margin density at zero** (gc14's own derived preflight, from the QA80 atlas): predicts the *magnitude* of realized d_seg change per unit weight-space excursion.
2. **NEW — the visible fraction φ.** From the completed `ms4d` bundle + the composite-R adjoint: the fraction of a uniform parameter-space kick whose image lies in `range(A)`. **This is the effect-size prediction the operator asked for**: D± should beat B by roughly **1/φ** in kick efficiency at matched ‖Δθ‖. If φ is near 1, D± cannot help and the arms should not run. If φ ≈ 0.2–0.5 (plausible given the resize nullity figures), D− predicts a **2–5×** efficiency gain and the experiment is strongly indicated. **Pre-registering: if the preflight returns φ > 0.8, I withdraw arms D± and the ranking in §5.5 changes.**

**Handoff to #815 R1 (amended to four-armed):** arms **B′, C, D−** plus the existing control; arm A requires the opt-state persistence fix (gc14 R5) and should be sequenced with it. Every arm is a **DSL `Lever` factory to be BUILT** — `lever_reset_operator(what=…, to=…, structure=…)` — never a hand-added trainer flag. None auto-fires; heavy/paid remains operator-GO.

---

## §8 CONDITIONAL-VALIDITY RE-GRADE: QA24 from-birth domination

**Standing verdict (b4s §1/§4):** "MEASURED-DOMINATED by warm ep641 (0.686 vs 0.608). Closed at INSTANCE."

**Three independent confounds, in increasing order of severity.**

1. *(the seed's)* **The D16 cap.** `--grid-downsample` choices={8,16}; the arm could not go coarser than the warm base. Real, but it does not by itself flip a 0.078 S gap.
2. **Byte mismatch.** bc1 253,858 B vs warm 273,004 B — the from-birth arm was **19,146 B cheaper = 0.01275 S ahead on rate**. The sg1 Contrarian's own falsifier was pre-registered "**at matched bytes**"; the comparison was never matched.
3. **COMPUTE — the binding one, and it was never stated anywhere.** bc1 was measured at **ep399**. The warm base is **ep641**. `--epochs` is the *cumulative* target in this trainer (window_01's ticket carries `--epochs 666` for a run that ends at ep665), so these are cumulative-from-birth epoch counts. **The warm arm had 60.7% more training than the arm it "dominated."**

**The matched-compute normalization.** The warm lineage has an n600 receipt at the nearest comparable point: **ep499, seg 0.49410 S** (gc13 §2.0 / qa92, carried in gc14 §5.1). Projecting bc1 forward 100 epochs at the warm lineage's **own** ep499→641 rate (−0.06769 S / 142 ep = **−4.767e-4 S/epoch**):

| | seg S | rate S | S_add |
|---|---:|---:|---:|
| bc1 fresh @ ep399 (MEASURED) | 0.51690 | 0.169034 | 0.68593 |
| **bc1 fresh @ ep499 (PROJECTED)** | **0.46923** | 0.169034 | **0.63826** |
| **warm @ ep499 (MEASURED seg; bytes back-estimated)** | **0.49410** | ~0.179 | **~0.67318** |
| warm @ ep641 (MEASURED) | 0.42641 | 0.181782 | 0.60819 |

**At matched compute the from-birth arm projects ~0.035 S AHEAD, not 0.078 S behind. The sign flips.**

**Why the projection is conservative (it understates fresh):** (i) the rate applied is from a *later, more converged* segment of the warm lineage; bc1 at ep399 sits at a *higher* d_seg (0.005169 vs 0.004941) and is therefore *less* converged, so its true local rate is faster — decay works in fresh's favour here. (ii) The warm lineage accumulated **more restart impulses** over its longer life (§5.1: 16.17 free epochs each), so part of the residual warm advantage is the reset bonus, not the warmness.

**Verdict:** `MEASURED-DOMINATED` → **`CONDITIONALLY RE-OPENED — the domination verdict is confounded by a 60.7% compute deficit and an unmatched byte basis; on the best available matched-epoch projection the sign reverses.`** `verdict_scope: INSTANCE` · `empirical_verification_status: INFERRED_FROM_DOMAIN_LITERATURE` → strictly, **DERIVED-BY-EXTRAPOLATION**, downgraded to `PROVISIONAL-PENDING-VERIFICATION` per the recursive-self-reflection protocol. **Verification (the only honest closure): re-measure the bc1 checkpoint at ep499+ OR compare at equal epoch.** That is a $0 telemetry read if bc1's gate series survives in custody, and a bounded continuation otherwise.

**What this does NOT license.** It does not say fresh wins. It says **we have never actually run the comparison**, and the record has been carrying a closure that the arithmetic does not support. That is exactly the janky-prototype-closure class the ANTI-SIGNAL-LOSS non-negotiable exists to re-open.

---

## §9 HYBRID CANDIDATES, RANKED (non-additive: raced, never summed)

| # | candidate | status | S-relevant prior | verdict |
|---|---|---|---|---|
| **1** | **Structured warm reset** (warm weights · fresh m · scorer-prior v · per-channel) | **NEVER RUN**; #725 built, ms4d complete | changes the *efficiency* of every impulse; φ-preflight predicts 1/φ | **PURSUE — rank 1.** $0 build on existing artifacts, ~0× wall-clock, tests the standing-law violation |
| **2** | **Bias-corrected reset (B′)** | never run; one field | isolates the mechanism outright | **PURSUE — rank 2** (cheapest decisive arm in the campaign) |
| **3** | **Warm-weights / fresh-optimizer** (what we accidentally ran) | **the incumbent**, measured −0.018303 S/window | gc14 geometric remaining 0.00946 S | **INCUMBENT — becomes the control, not a candidate** |
| **4** | **Fresh-with-solve-INIT-tokens** | `solve_project` **ADOPTED at −28.9%** (sc2 §14, n600 matched-epoch) | the campaign's ONE measured init lever; already ON in every current config incl. bc1 | **ALREADY COMPOSED** — not a new candidate; it is the reason a fresh birth is cheaper than it used to be, and it is a receipt that **knob-2-style priors pay** (a *token* prior bought 28.9%) |
| **5** | **QA84 rowband from-birth** | **BLOCKED-measured**: no D8 parent exists anywhere | pre-registered seg 0.431 via Lane reach-limit; theorem-certified band [160,240)/1248 | **PURSUE — the #1 genuinely fresh-only cell** (§10). Structural necessity, not preference |
| **6** | **KD-from-warm-into-fresh** (#74/#129 `kd_warm_start_dir`) | **BUILT, 6 NO-FAKE tests, DEFAULT-OFF, never fired on tr1** | designed exactly to carry a basin across a shape change | **PURSUE — rank 3, and it is the DIRECT ANSWER to "reset against upstream weights"**: it is knob-2-at-the-weight-level, already built. Its blocker was always "needs from-scratch"; the rowband cell **supplies** that from-scratch run, so #6 and #5 compose into ONE run |
| **7** | **from-birth-KD** (b4s DEFERRED) | dw1: continuation-KD CLOSED at FORMULATION; from-birth "still live, deprioritized" | falsifier: beat rung-1 at matched compute | **MERGE INTO #6** — they are the same cell; the actuator for #7 is #6, which is already built. Recording this as a **found orphan**: a DEFERRED cell whose actuator was built 6 weeks earlier and never connected |
| **8** | **Periodic hard restarts (SGDR/SWA)** | `1608.03983` logged, **never fired**; measured caution: "a FULL restart (1.0×) reproduced the v3 destabilization" | LR-schedule restarts, orthogonal to moment resets | **HOLD** — we are *already* getting an un-scheduled 6.57× spike; adding a second, LR-side restart on top is uncontrolled. Revisit only after B′ prices the existing one |
| **9** | **Fresh-with-rate-in-loss-from-birth** | `--rate-model entropy --w-rate 0.05` is **already in-loop** and inherited | gc14: rate drifted +4,055 B/window | **RE-SCOPED, and the seed is corrected**: rate is *not* an afterthought — it has been in the loss since ep641's config. The real finding is that **w_rate has been frozen at 0.05 for the whole lineage under constants-are-poison discipline**, so rate is in-loop but **un-tuned**. That is a *warm* sweep, not a birth |
| **10** | **Weight rewinding / lottery-ticket** | measured destabilization receipt (v3, full restart) | — | **NAMED, NOT FIRED** |

---

## §10 THE FRESH-RUN DECISION FUNCTION (pre-registered)

**A from-birth run fires if and ONLY if it targets something warm cannot structurally reach.** Three named triggers; **any one suffices**; all are pre-registered.

```
FIRE-FRESH  ⟺  T1 ∨ T2 ∨ T3

T1  STRUCTURAL-IMPOSSIBILITY (the strong trigger)
    A named, BUILT, byte-relevant grammar has NO compatible parent checkpoint anywhere.
    CURRENTLY TRUE — exactly once: QA84 rowband.
      receipt: pa1r "NO D8/rowband checkpoint exists anywhere (every tr1 ckpt is D16)"
      artifact: RowBandGrammar BUILT (ddm_b2b e8d531e735), --token-rowband-spec exists,
                DOF 1248, band [160,240)/1248 theorem-certified rate-optimal at the >=50% gate
    Falsifier: a D8-compatible parent is produced by any other route -> T1 lapses to a warm tail.

T2  MATCHED-COMPUTE REVERSAL (the §8 trigger)
    A from-birth arm previously closed as dominated is shown, at MATCHED epoch and MATCHED
    bytes, to be within 0.02 S of the warm lineage or ahead.
    CURRENTLY: PROVISIONAL-TRUE for QA24 (projection: fresh ahead ~0.035 S).
    Fires only on the re-measurement, never on the projection.

T3  PATH-DEPENDENCE CEILING (the operator's seed item 2, made falsifiable)
    From the ru1 atlas: let E = fraction of erased GT components lying in cells the current
    warm trunk has NEVER re-birthed at any gate of the burn.
      E >= 0.30  -> birth buys real, unreachable structure  -> T3 TRUE
      E <  0.30  -> warm reaches it; the ERF in-loop receipt (567->508, Lane -53) governs -> FALSE
    CURRENTLY: UNMEASURED. This is a $0 read from banked artifacts and it is the single
    measurement that converts the operator's strongest pro-fresh intuition into a number.
```

**If it fires, the EXACT config** — one run that composes every fresh-only item so we never pay the birth cost twice:

```
BASE      : tr1 from scratch, seed118 lotto, --num-pairs 600, --batch-pairs 8, --lr 2e-3
GRAMMAR   : --token-rowband-spec (D8 base, bulk 2x2 TIED, flip-band rows 160-240 FREE)   [T1]
INIT      : --token-init-mode solve_project                       [sc2 -28.9%, ADOPTED]
CARRY     : kd_warm_start_dir <- the ep641/ep946 warm endpoint     [#74/#129, BUILT, never fired]
            => the fresh run INHERITS the warm basin instead of discarding it (#6 x #7 merged)
RATE      : --rate-model entropy --w-rate 0.05 (unchanged; constants-are-poison)
SEG       : --class-weight-lane 1.3 (R6, inherited) + the DESIGNED-STUB forces ONLY IF
            trainer-wired first (R1/R2/R3/R8/R14 are stubs -- do NOT claim a from-ep0 stack
            that does not exist; wiring is the blocker, not birth -- see §4 item 2)
RESET     : the §7 winner (never the accidental zero-reset)
P0        : resumable-from-disk, per-stage EMA checkpoints, DSL-hashed ticket, memory preflight,
            governed launcher, argv-diff pre-fire assert
```

**Price, and the two lenses the mandate names.**
- **Cost:** bc1's receipt is **400 epochs / 480 min**; a rowband D8 run has 1,248 DOF vs D16's 768, so budget **~1.6× ⇒ ~13 h** for a 400-epoch equivalent, or ~2 governed overnight sessions. $0 cash (local MLX), one scorer slot at the endpoint.
- **Schmidhuber lens:** on a decade horizon 13 h is nothing, and the run generates *compression progress* on an axis (rate grammar) where the derivative has not yet collapsed — unlike seg-continuation, whose derivative gc14 measured collapsing 3.2× in one step. **On his own criterion the fresh rowband run is the higher-drive domain.** He endorses.
- **Carmack lens:** only if it buys what warm can't. **T1 says it literally cannot be bought otherwise** — there is no D8 parent in existence. He endorses T1, and *only* T1. He explicitly refuses a fresh run motivated by T3 alone until E is measured, and refuses one motivated by "the stack has never run from ep0" while five of the six forces are unbuilt stubs.

**Both lenses converge on: fire fresh for the rowband grammar, carry the warm basin in via the built KD actuator, and do not fire fresh for the seg axis.**

---

## §11 BACKCAST from the bar (through both options)

Bar = `min(0.15, official 0.172141)`. Own-vehicle exact-protocol line = **0.9639878**; gap to bar = **0.791847**.

| path | mechanism | S it can deliver | gap closed |
|---|---|---:|---:|
| **warm continuation, as-is** | 1 impulse/window × landscape r=0.310 | **0.00946 S** (gc14 geometric total) | **1.2%** |
| warm + 2× cadence (R1) | 2 impulses/window, concave | ≤0.019 S | ≤2.4% |
| **warm + structured reset (D−)** | same impulses, 1/φ efficiency | **0.009–0.05 S** *if* φ ∈ [0.2, 0.8] — **UNMEASURED, φ-preflight owed** | 1–6% |
| **fresh rowband (T1)** | new rate grammar, theorem-certified band | pre-registered **seg 0.431** target; rate delta unpriced | unpriced |
| **RATE — QA24 cell_drop50 (banked)** | byte-closed `a6398e44` | **−0.098 S** | **12.4%** |
| **RATE — wr1 #766 Knee-A/B (banked)** | 274,333 B / 174,578 B | −0.197 / −0.263 S | 25% / 33% |

**Backcast verdict, unchanged from gc14 and reinforced:** neither pole of the operator's question is a bar-reaching path on the seg axis. **The reset-metric work is justified as the cheapest efficiency multiplier on an axis we are running anyway ($0, ~0× wall-clock), not as a gap-closer.** The gap is RATE — and the fresh question's one strong trigger (T1 rowband) is *itself* a rate cell. That is the reconciliation of the operator's two questions with gc14's B5-C default: **hand the slot to rate, and the fresh run you fire there is the rowband one.**

---

## §12 SANDWICH (upper leg: what a reset can reach / lower leg: what bounds it)

| leg | quantity | value |
|---|---|---:|
| **upper — per-boundary impulse** | ∫(η−1)dt, derived | **1,212.6 sign-steps = 16.17 epochs** (fixed, cadence-independent) |
| upper — realized per boundary | gc14 measured step w02→w03 | −1.118e-4 gate d_seg |
| **efficiency multiplier available** | 1/φ from the composite-R adjoint | **UNMEASURED — the binding unknown; $0 preflight owed** |
| lower — landscape return | gc14 measured r | **0.310** per window |
| lower — reachable seg pool | fl1 above-floor (Undriv + Movable) | **0.0258 S**, ~81% consumed |
| lower — hard floor on this leg | ru1 GT-jitter-typed reachable | ~6e-4 ≈ corner-C |

**Sandwich reading:** the impulse is **known and constant**; the landscape return is **measured and collapsing**; the only term with headroom is the **efficiency multiplier 1/φ**, and it is the one term nobody has measured. **That is precisely why knob 2 outranks knob 0 in §5.5** — it is the only leg of the sandwich that is still open.

---

## §13 EXHAUST — pool census with Contrarian P·O bounds (SKIP rule: P·O < 0.05 S)

| pool | P·O (S) | status |
|---|---:|---|
| seg continuation, current cadence | 0.00946 | **SKIP** (gc14's verdict, unchanged) |
| cadence 2× (R1 as originally scoped) | ≤0.019 | **SKIP on S; PURSUE on information** — but now demoted to a *control arm* |
| **reset metric (knob 2, D±)** | 0.009 × (1/φ) ⇒ **0.011–0.047** | **BELOW the bound on the central estimate — PURSUE ANYWAY, on three grounds the bound does not price**: (i) it is a **standing-law violation** (generic metric, underived) and closing that is not optional; (ii) it is **$0 build + ~0× wall-clock** on artifacts already built; (iii) B′ alone **re-prices the campaign's only measured descent**, which is decision-relevant far beyond its own ΔS. **Contrarian's bound is respected and explicitly overridden with reasons.** |
| **bias-correction isolation (B′)** | ΔS ≈ 0, **information ≈ everything** | **PURSUE — the highest information-per-dollar item in the campaign** |
| **fresh rowband (T1)** | unpriced; pre-registered seg 0.431 | **PURSUE** — the only structurally-fresh-only cell |
| QA24 from-birth re-measure | reverses a 0.078 S mis-closure | **PURSUE at $0** if bc1 telemetry survives |
| **RATE — cell_drop50 / wr1 #766** | **0.098 / 0.197 / 0.263** | **LIVE, largest pools; unchanged from gc14** |
| post-hoc injection | 0.0171, ERF net-worse | **DEAD** |

**Census verdict:** the census still routes **B5-C (hand the seg slot to RATE)**. gc15 does not overturn gc14; it adds **two $0 riders** (B′ and the φ-preflight) that ride along regardless of which axis owns the slot, plus **one fresh run (T1 rowband) that is itself on the rate axis.**

---

## §14 RESEARCH (papers-checked ledger consulted FIRST; named consumers; what transfers)

**Consulted first, in-repo:** `oss_untried_technique_candidates_for_synergy_pass` §5c (SGDR `1608.03983` — logged, ranked, **never fired**, with our own antagonism flag: *"a restart can un-place an already-correct coarse partition if fired too late"*) · `curriculum_openpilot_seeded_deepmath_dsl` (**measured**: full 1.0× restart reproduced v3 destabilization ⇒ partial 0.1× floor) · `negative_audit_wave_20260713` row 12 (**K-FAC DISCARD**) · `sy1` S1-POLICY (Fisher mirror/K-FAC named as a *policy mode*, Euclidean default REFUSED) · `src/tac/canonical_equations/adam_v_variance_warmup_20260717.py` (**`adam_v_variance_warmup_length_v1` — already registered**).

**External, this session:**

| source | what transfers | named consumer |
|---|---|---|
| **"Simplifying Adam: Bias Correction Debunked"** (arXiv 2511.20516) — *"Bias correction is not a true performance enhancer; but merely an implicit, and often clumsy, learning-rate warm-up"*, and it **"non-negligibly alters the effective learning rate for default settings (0.9, 0.999)"** | **Independent confirmation of §5.1 at exactly our β.** Their framing (bias correction ≈ a warm-up schedule) is the same object as my η: omitting it *removes* the implicit warm-up and lets early steps run 3–6.6× hot. It also tells us this is a **known, characterized artifact**, not a novel discovery — which raises, not lowers, our obligation to have priced it | **arm B′**; the §5.1 derivation |
| **"Why Warmup the Learning Rate? Underlying Mechanisms"** (NeurIPS 2024, arXiv 2406.09405) + **"Analyzing & Reducing the Need for LR Warmup in GPT Training"** — Adam's update ℓ₂-magnitude spikes at start, **"primarily stemming from β₁ bias correction"**; removing it mitigates artificially large initial updates | Confirms the *family*; note the **scope difference I must state**: their spike is in the correct-m/uncorrected-v regime (≈31.6× at t=1 by the same algebra); ours is the **neither-corrected** regime (3.16× → 6.57×). Same mechanism, different magnitude. **Do not quote their effect size as ours** | §5.1 scope note; the η table is OURS |
| **SGDR** (Loshchilov & Hutter, arXiv 1608.03983, ICLR 2017) + **"When to restart? Escalating restarts on convergence"** (arXiv 2603.04117) | The mechanism question is **NOT settled in the literature** — restarts are credited with **both** exploration (escape narrow minima, reach flatter regions) **and** conditioning (accelerate on ill-conditioned problems). **This is why §7 must be multi-armed**: we cannot resolve it by citation, and neither can they | §7 arm design; the H1/H2 split |
| **Lottery ticket / weight rewinding** | The instance of knob 2 at the **weight** level. Transfers as a *named cell*, not a config — and our own v3 destabilization receipt is the stronger local evidence | §5.3(2c), held |
| **K-FAC / natural gradient / mirror descent** | The conditioning arm's theory. **Our own DISCARD (07-13) is re-opened at FORMULATION scope**: its stated grounds were "does not change the frozen teacher-call bottleneck" (a *wall-clock* objection) and "no unique current ticket consumer" (a *routing* objection). A **boundary insertion** costs ~0 wall-clock and now has a consumer. **Neither original objection survives the new formulation** | §5.3(2b); arms D± |

**Honest research verdict:** no external source hands us a config. Two of them independently confirm our mechanism at our exact β, one tells us the mechanism question is genuinely open, and our own ledger already contained SGDR-logged-never-fired and a measured full-restart destabilization. **The transferable content is the question structure, not the answer** — consistent with the no-old-lineage discipline applied to optimizers.

---

## §15 DERIVE-ORIGINAL (from the frozen scorer's structure, not a transplant)

**The boundary-impulse efficiency law.** Realized d_seg is a **count of sign flips of a margin field**, and the SegNet head is **exactly rank-4** with flip distance `d = |m| / ‖Δw_e‖` (measured; hb1 confirms rank 4 with singular values 4.703/2.831/2.039/2.018). A reset impulse displaces parameters by `Δθ` with the derived norm `‖Δθ‖ ≈ lr · √(I·N)` in sign-step units. The realized d_seg change is then

```
E[Δd_seg]  =  (margin density at 0)  ×  ‖ J_composite · Δθ ‖   ×  (per-class capacity weighting)
                     ^ QA80 atlas          ^ = φ·‖Δθ‖               ^ #725 cap_b(i)
```

Every factor on the right is **already measured or measurable at $0 from banked artifacts**. Three consequences that are ours, vehicle-native, and testable before any arm runs:

1. **The impulse is fixed; only φ and the capacity weighting are controllable.** So the reset-metric lever's entire value is `1/φ` — which makes the φ-preflight (§7) not a nicety but **the gate on whether arms D± should run at all**.
2. **The optimum weighting is class-dependent**, because #725 measured capacity concentration to be strongly class-pair-dependent (Lane–Movable 70.7% in 3 channels vs Road–MyCar 16.1% top channel). **A single global preconditioner is provably sub-optimal against our own measurement** — which composes exactly with fl1's per-class debt ranking (Lane #1, 13.1× corner-C).
3. **A prediction we can falsify before running anything:** the boundary step's per-class decomposition should be **proportional to per-class margin density at zero**, *not* to per-class residual S. gc14's endpoint bundle (`ddm_b4r_endpoint_extras.py`) produces per-class n600 deltas at ~18:40Z. **Pre-registering: if the w02→w03 boundary step's per-class split correlates with margin-density-at-zero (Spearman ρ > 0.6, n=5) rather than with residual level, §5.1 gains a second independent confirmation from data it was not fitted to.** n=5 is small and I say so; this is corroboration, not proof.

---

## §16 SUGGEST / CONSIDER / WONDER

**Suggest.** Make the reset operator a **first-class, declared, DSL-held object**. gc14 found the *window length* was a silent unpriced hyperparameter; gc15 finds the *reset semantics inside it* were too — and worse, they were set by a third-party library default. Every window decision record should carry `reset_event {what, to, structure, eta_profile_id, bias_correction}` alongside gc14's proposed `boundary_event`.

**Consider.** The trainer's own comment says the fresh moments implement "warm-start re-anchor law #517/#518." **The law was honored; its magnitude was never computed.** This is a distinct and more insidious failure class than an un-labelled constant: *a correctly-cited, deliberately-implemented law whose realized effect size nobody ever derived.* Sister to gc14's window-length finding, and I propose it as a named law: **"a cited law with an underived effect size is an unpriced lever."**

**Wonder (unresolved, and it is the biggest one).** If a two-character library default has been supplying the campaign's only measured seg descent, **what else in the stack is a third-party default doing load-bearing work?** β=(0.9, 0.999) itself is unswept. `eps=1e-8` sets the floor below which the sign-step degenerates. The `--lr 2e-3` was set once. gd1's generic-default census (QA82) covered *our* defaults; **it did not cover the defaults of our dependencies.** That is a named gap and a $0 census.

**A second wonder, on the fresh question specifically.** The operator's instinct that "protect-from-birth ≠ protect-after-erasure" is *structurally* right and I do not want the §4 correction to bury it. What makes it not-yet-actionable is that the protection stack is five-sixths unbuilt. **If the stack were built, the from-ep0 question would become the sharpest experiment in the campaign** — because then, and only then, "never erased" and "recovered" would be distinguishable states of the same vehicle. **Building the stack is therefore a prerequisite the fresh question should be allowed to pull forward.** That is a real, operator-visible re-prioritization: the DESIGNED-STUB forces (R1/R2/R3/R8/R14) are gating a question the operator wants answered.

---

## §17 WHAT I COULD NOT DO / OWED

- **No scorer jobs** (window_03 owns the slot). No per-class n600, no φ measurement, no arm fired. Everything here is source-derived, closed-form, or projected from banked receipts.
- **φ is UNMEASURED** and it gates arms D±. The preflight is $0 but needs the composite-R adjoint from the ms4d bundle — a read, not a run. **Owed before D± is designed further.**
- **The η profile is an upper envelope** under the constant-g idealization (§5.1). The realized multiplier is between 1 and η. This is stated, not hidden, and arm B′ measures it directly.
- **§8's matched-compute reversal is a PROJECTION**, cross-lineage, with the extrapolation rate named. It re-opens a closure; it does not establish fresh superiority. Downgraded to `PROVISIONAL-PENDING-VERIFICATION`.
- **T3's E (path-dependence ceiling) is UNMEASURED.** It is the number that would settle the operator's strongest pro-fresh intuition, and I could not run it in this slot.
- **My own correction, recorded:** I initially wrote that a zero-reset makes the first step *smaller* (the standard "biased toward zero" intuition). **Wrong** — v's bias (1−β₂)=0.001 under a square root dominates m's (1−β₁)=0.1, giving a **larger** step by 3.16×. I caught this by deriving rather than recalling, and the external literature confirms the corrected direction. A memo built on the wrong sign would have recommended the exact opposite intervention.
- **I did not verify that bc1's ep399 checkpoint or gate series still exists in custody**, which determines whether §8's verification is $0 or requires a run.

---

## §18 TYPED VERDICTS (per-landing pantheon review)

| # | claim | scope | evidence grade | verdict |
|---|---|---|---|---|
| V1 | MLX `Adam` defaults `bias_correction=False`; tr1 does not override | STRUCTURAL | `inspect.getsource` + trainer L1543 | **MEASURED (source)** |
| V2 | Post-reset effective-LR multiplier η(t)=(1−β₁ᵗ)/√(1−β₂ᵗ); η(1)=3.162, peak η(12)=6.569 | STRUCTURAL | closed form + numeric | **DERIVED** |
| V3 | Each boundary injects 1,212.6 extra sign-steps = 16.17 epochs = 11.5%/window; 81.7% in first 13 ep | INSTANCE (this config) | derived at measured argv (batch 8, 600 pairs, lr 2e-3) | **DERIVED** |
| V4 | η predicts all four of gc14's boundary-step observations without being fitted to them | FORMULATION | §5.1 | **DERIVED** |
| V5 | η is the *cause* of the measured d_seg step | — | — | **INFERRED — arm B′ owed** (upgrade of gc14 V5 from INFERRED-with-no-mechanism to INFERRED-with-closed-form) |
| V6 | The zero-reset is a generic/undeclared metric, violating `generic_basis_metric_never_optimal` + sy1 S1-POLICY | STRUCTURAL | code + standing law | **DERIVED** |
| V7 | tr1 exposes 0 of 64 flags governing any reset knob | STRUCTURAL | flag census | **MEASURED** |
| V8 | QA24 from-birth was measured at ep399 vs warm ep641 (+60.7% compute) and at unmatched bytes | INSTANCE | b4s §2/§4 receipts | **MEASURED** |
| V9 | At matched epoch 499 the from-birth arm projects ~0.035 S AHEAD (sign reversal) | INSTANCE | extrapolation at the warm lineage's own rate | **DERIVED-BY-EXTRAPOLATION — PROVISIONAL-PENDING-VERIFICATION** |
| V10 | QA84 rowband is structurally fresh-only (no D8 parent exists) | STRUCTURAL | pa1r / QA84 ledger | **MEASURED** |
| V11 | The #74/#129 KD-warm-start actuator is BUILT, tested, never fired, and is the actuator for the DEFERRED from-birth-KD cell | STRUCTURAL | kd_warm_start_actuator memo + b4s §3b | **MEASURED — orphan found** |
| V12 | A structured warm reset dominates both poles | FORMULATION | §5–§9 reasoning; not yet measured | **DERIVED (argument), UNMEASURED (effect)** |
| V13 | Knob 2 outranks knob 0 (cadence) in information per $0 | FORMULATION | §5.5 + §12 sandwich | **DERIVED** |
| V14 | K-FAC DISCARD (07-13) is conditionally re-opened at FORMULATION scope | FORMULATION | both stated grounds fail for a boundary insertion | **RE-OPENED** |
| V15 | Fresh birth is required for the seg axis | — | §4 item 2, §10 | **NOT ESTABLISHED — refused** |
| V16 | j4's β₂-derived 2000-step ramp and my 1/(1−β₂)=1000-step timescale are the same object, derived independently | STRUCTURAL | j4 FEED + §5.1 | **DERIVED — two-route corroboration** |
| V17 | Pointer moved | — | — | **NO. `0.1910828242` [contest-CPU] UNMOVED.** |

**Relative significance (pace ≠ direction).** The largest number here is **not** any ΔS — it is **0.098 S** (QA24 cell_drop50, banked, on the rate axis), unchanged from gc14 and still 5–10× anything on this convocation's axis. gc15's own S-contribution is **plausibly 0.011–0.047 S and currently UNMEASURED**. Its real product is **a re-pricing of the campaign's only measured descent** and **a $0 arm (B′) that tells us whether that descent was an artifact.** [magnitude-ok on dismissals below 0.001 S.]

---

## §19 DISSENT (verbatim, preserved)

- **Schmidhuber (LEAD):** "I asked last time for the second derivative and got r = 0.310. Now I am handed a first-principles reason why: a fixed impulse against a decaying landscape. Good. But I dissent from the celebration. If the descent is a bias-correction artifact, then the *compression progress* we have been measuring was never model improvement — it was the optimizer falling downhill faster because nobody normalized its step. My drive formulation cares about *predictable* progress; an artifact is the least predictable kind. Run B′ before anything else, and if the step vanishes, say loudly that the month produced no learning-driven descent at all."
- **Contrarian:** "Two objections. First, η is derived under constant g. Real gradients are noisy, and noise *reduces* m/√v — possibly a lot. The memo admits this as an 'upper envelope' and then uses the envelope numbers in every table anyway. 16.17 epochs is an upper bound presented as a quantity. Second, §8's reversal leans on projecting a *different lineage's* decay rate onto bc1 across 100 epochs. That is exactly the cross-instrument extrapolation Schmidhuber objected to in gc14, now used to overturn a closure rather than to defend one. Apply the standard symmetrically."
- **Assumption-Adversary:** "The memo's own framing assumes the reset's *value* is its displacement. But displacement is not learning. A 6.57× step in a well-conditioned basin is just a bigger step — it goes further along the *same* trajectory, arriving where the run would have arrived later. If that is what is happening, then the boundary step is not a gain at all, it is **time-travel**: the window boundary buys 16 epochs of ordinary descent for free, and gc14's r = 0.310 is then just the *ordinary* decay of ordinary training, sampled at boundaries. Under that reading, cadence gives you nothing that epochs would not, and the whole 'restarts are a lever' framing collapses. **Nothing in this memo distinguishes that hypothesis from H1.** Add it as H5 and design for it."
- **Rudin (CO-LEAD):** "The decision function is readable and every branch names its receipt — I endorse the form. But T3's threshold, E ≥ 0.30, is a bare constant with no derivation, in a memo whose §2 arraigns the campaign for exactly that. Derive it or race it."
- **Yousfi:** "The φ preflight is the right instrument and it is being treated as a rider. It is not a rider. If φ is small, then most of every step this campaign has ever taken — not just the reset kick, *every step* — has been spent in directions the scorer cannot see. That is a statement about the entire training history, not about boundaries. Measure φ first and read it as a campaign-level diagnostic."
- **Hotz:** "One field. `bias_correction=True`. Two hours. You either still have a descent or you don't. Everything else in this memo is downstream of that one bit and should wait for it."
- **Time-Traveler:** "You already had this. `adam_v_variance_warmup_length_v1` has been a *registered canonical equation* since 07-17, and j4 derived a 2000-step β₂ ramp from it on 07-23. The apparatus knew the β₂ timescale governs the post-reset window. Two convocations then read a boundary step and called it learning. We do not need new information. We need to read what we have already written down."
- **Shannon:** "A sign step is the maximum-entropy step: uniform magnitude, direction carrying one bit per coordinate. We have been injecting maximum-entropy perturbations into a system whose metric we have measured to completion. That is not a subtle inefficiency; it is discarding the entire channel."

---

## §20 ASSUMPTION-ADVERSARY VERDICTS (`empirical_verification_status`)

| assumption | classification | status | rationale |
|---|---|---|---|
| "The fresh optimizer at a boundary is a neutral restart" | **CARGO-CULTED** | `VERIFIED_VIA_SOURCE_INSPECTION` | It is a 3.16–6.57× LR excursion. Falsified this convocation. |
| "Adam's library defaults are a safe backdrop" | **CARGO-CULTED** | `VERIFIED_VIA_SOURCE_INSPECTION` | β and `bias_correction` are unswept third-party defaults doing load-bearing work. |
| "A zero reset is the neutral choice for `v`" | **CARGO-CULTED** | `VERIFIED_VIA_SOURCE_INSPECTION` | Zero = the generic metric; forbidden by our own 07-29 law absent derivation or race. |
| "QA24 from-birth was measured-dominated" | **CARGO-CULTED** | `VERIFIED_VIA_EMPIRICAL_ANCHOR` (the epoch counts) | Unmatched compute (+60.7%) and unmatched bytes; the sign reverses on normalization. |
| "The protection stack has never run from ep0" | **HARD-EARNED but MIS-FRAMED** | `VERIFIED_VIA_SOURCE_INSPECTION` (b4s §3b) | True — but because it is *unbuilt*, not because runs were warm. Wiring is the blocker. |
| "Restart benefit is exploration OR conditioning" | **CARGO-CULTED (false dichotomy)** | `ASSUMED_AWAITING_VERIFICATION` | Literature credits both and does not settle it; the Assumption-Adversary adds a third (H5: pure time-travel). §7 must carry all three. |
| "The reset displacement is a gain" | **CARGO-CULTED** | `ASSUMED_AWAITING_VERIFICATION` | H5. Nothing here distinguishes gain from time-travel. **Downgraded accordingly in §21.** |
| "η's constant-g form gives the realized multiplier" | **CARGO-CULTED** | `ASSUMED_AWAITING_VERIFICATION` | Upper envelope only; Contrarian's objection sustained. |
| "The scorer metric is in custody and usable" | **HARD-EARNED** | `VERIFIED_VIA_EMPIRICAL_ANCHOR` | ms4d BUNDLE-COMPLETE passes the strict MS3 loader; #725 argmax custody 0.999999991522895. |
| "fl1 floors are FORMULATION-scoped" | **HARD-EARNED** | `VERIFIED_VIA_EMPIRICAL_ANCHOR` | fl1 scopes itself; unchanged. |

---

## §21 VERDICT STATUS

**`PROCEED_WITH_REVISIONS`.** Per the recursive-self-reflection protocol, three claims are downgraded:

- **§5.1's magnitude table is `PROVISIONAL-PENDING-VERIFICATION`** on the Contrarian's constant-g objection. The *mechanism* (V1, V2, V4, V6) is not downgraded — it is source-derived. Only the realized 16.17-epoch figure is provisional. Verification: arm B′.
- **§8's matched-compute reversal is `PROVISIONAL-PENDING-VERIFICATION`** on the cross-lineage extrapolation, applying Schmidhuber's gc14 standard symmetrically as the Contrarian demands. It is sufficient to **re-open** the closure, never to reverse it.
- **The "displacement is a gain" premise is `ASSUMED_AWAITING_VERIFICATION`** on the Assumption-Adversary's H5 (time-travel). **§7 is amended to carry H5**, whose discriminator is: under H5 the boundary step should be *exactly reproducible* by ~16 extra epochs without a reset. That is a fifth arm and it is nearly free — it is arm A run 16 epochs longer.
- **T3's E ≥ 0.30 threshold is flagged as a bare constant** per Rudin and must be derived or raced before T3 can fire.

Not downgraded: V1, V2, V4, V6, V7, V8, V10, V11, V13, V16 — all source-derived or measured.

---

## §22 THE ONE-PARAGRAPH ANSWER TO THE OPERATOR

**Neither pole — and the reason is that the "reset" we have been doing every 140 epochs is not a neutral restart but an unpriced 3.16–6.57× learning-rate spike, because MLX's Adam defaults `bias_correction=False` and our trainer never overrides it; each boundary therefore injects a fixed ~1,212 extra sign-steps ≈ 16 epochs of free parameter displacement, 82% of it in the first 13 epochs, which derives gc14's entire boundary-step shape from source with no new experiment and makes it likely that the campaign's only measured seg descent this month is an optimizer artifact rather than learning.** Given that, warm already beats fresh on the mechanics — a warm run with k windows gets k of those kicks *and* keeps the path, while a fresh birth gets one kick and must re-learn everything — so **do not birth a fresh run for the seg axis**; but the restart count is a *knob*, not a property of warmness, and the far bigger unexploited lever is **what we reset *to***: right now it is zero, which makes the post-reset step a uniform sign step — the maximally generic, metric-free direction — in direct violation of our own standing law that a generic metric is forbidden without derivation or a race, at the one point in training where inserting our fully-custodied frozen-scorer metric (ms4d complete, #725 per-channel capacity built and measured, rank-4 head exact) costs **zero extra wall-clock because we are discarding the metric anyway**. **So: a structured reset on a warm run plausibly dominates both poles, and that is the answer** — with two caveats I will not soften: the benefit may *be* the disorder (so the metric arms must be magnitude-matched and vary only direction), and it may be pure time-travel (a bigger step arriving sooner where the run was already going), and one $0 two-hour arm — `bias_correction=True`, everything else identical — settles which. **Fresh birth stays justified for exactly one thing, and it is on the rate axis, not the seg axis: the QA84 rowband D8 grammar, which is BUILT and for which no compatible checkpoint exists anywhere, so it cannot be reached warm at all — and when we fire it we should carry the warm basin in through the #74/#129 KD-warm-start actuator that was built six weeks ago, tested, and never once fired.** Separately, the record needs correcting: the QA24 "from-birth is measured-dominated" closure compared a fresh arm at epoch 399 against a warm arm at epoch 641 — 60.7% more training — and at matched epoch the sign reverses, so that closure is re-opened. The pointer is **`0.1910828242` [contest-CPU], UNMOVED**; nothing here is a score, and the gap is still RATE.
