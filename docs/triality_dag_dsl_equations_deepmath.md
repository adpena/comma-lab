# The Triality — DAG ↔ DSL ↔ equations (comprehensive deep-math + meta reference)

> **Pointer target.** CLAUDE.md carries the concise awareness block; this file is the
> comprehensive version. Canonical memory: `project_witness_dsl_and_dag_dsl_duality`.
> Sisters: `unified-variational-levelset-flow-everything-is-facets`,
> `all-automated-worldclass-recursive-review-deepmath-confirmed`,
> `asbuilt-maintenance-loop-and-adversarial-experiment-gate`.
> Guardrail: everything here is MEANS. The pointer (0.19110) moves only through a
> byte-closed `upstream/evaluate.py` n600 exact row (CPU/CUDA, never MPS).

## 1. What "triality" means here (the Spin(8) analogy, made precise)

In representation theory, **Spin(8)** is unique: it has three inequivalent 8-dimensional
irreducible representations — the vector **8v**, the spinor **8s**, the cospinor **8c** — and
its outer automorphism group is **S₃**, which *permutes the three*. They are not "the same
representation written three ways"; they are genuinely distinct objects that describe **one
group**, cyclically interchangeable. That cyclic three-fold symmetry is **triality**.

Our campaign has the same structure. The compression program is **one object** seen through
**three genuinely different representations**, cyclically related:

| Leg | Representation | Where it lives | What it is good at |
|---|---|---|---|
| **DAG** | trajectory / history | `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_*.md` (FEED-* blocks) | *what happened* — the measured path, ordering, provenance |
| **DSL** | executable program | `tac.witness_dsl.{curriculum_dsl, gauge, campaign}` → trainer CLI argv | *what to do next* — compiles intent into a deterministic run |
| **equations** | the law / math | `tac.canonical_equations` (registry + `EmpiricalAnchor`) | *why it works* — the derived, confirmed relationships |

No leg is primary. A finding is only "known" when it is expressible in all three and they
**agree**. Drift between legs (a DAG claim with no equation; a DSL flag with no DAG row; an
equation no run produced) is the campaign-level form of forgetting.

## 2. Two layers of one-object (the double unification)

There are actually **two** one-objects, nested — and that nesting is the "Understand"-mind
structure the operator named:

- **Layer A — the witness (physics facets).** The witness is ONE variational level-set flow
  of a Morse–Smale complex; distortion (Fisher/margin/UNIWARD), representation (curvelet basis
  = rate/MDL), curriculum (coarse→fine scale = persistence order = temperature annealing),
  dimensionality (complex DOF ~8 → Whitney ~17–19), temporal/pose (the se(3) ego-screw), and
  compute (MLX-first) are all **facets of that one flow**, not separate levers. Canonical:
  `unified-variational-levelset-flow-everything-is-facets`.

- **Layer B — the campaign (representational views).** The *process* of building the witness is
  ALSO one object, seen through the three triality legs above. DAG/DSL/equations are three views
  of the campaign the way distortion/representation/curriculum are three facets of the witness.

The same "one object, many consistent views" pattern at both layers — the witness *and* the
program that produces it — is the double unification. Keeping both internally consistent is the
whole discipline.

## 3. The cycle engine (the triality is dynamic, not static)

Triality is not a filing scheme; it is a **cycle** that compounds each pass:

```
   DAG ─────────▶ DSL ──────────▶ run ──▶ measured n600 rows ──▶ equations ──┐
 (trajectory)  (decide + compile)                              (confirm law)   │
     ▲                                                                          │
     └───────────────────── next DAG FEED row ◀────────────────────────────────┘
```

1. **DAG → DSL (decide):** read the measured trajectory; `campaign.decide_next_stage`
   (`StagePolicy`: `plateau_abs_slope`, slope-sign convention) → EXTEND / ADVANCE / RERUN /
   leap-residual; `plan_adaptive_step` emits the next `WitnessProgram` → trainer argv. Pure
   function of (on-disk trajectory, policy) ⇒ deterministic + reproducible.
2. **DSL → run:** the emitted argv trains the witness (resumable, per-stage checkpoints, EMA
   shadow). CONTAINMENT: the DSL *emits* argv; it never auto-fires a heavy/paid launch.
3. **run → rows:** `campaign.harvest_arm` reads `run.log` verdicts → measured d_seg trajectory
   (n600 authority, byte-closed through `tools/levelset_byte_close_and_eval.py` for any
   load-bearing verdict).
4. **rows → equations (confirm):** each signal is *derived and registered* against
   `tac.canonical_equations` with an `EmpiricalAnchor` — not eyeballed.
5. **equations → next DAG:** append a FEED row; the confirmed law informs the next `decide`.

This cycle **is** the automated witness instrument (task #216). Hand-cranking any leg (polling
the log by eye, pasting a number into chat, choosing the next stage by feel) is the anti-pattern
the automation quality-bar extincts (`all-automated-worldclass-recursive-review-deepmath-confirmed`).

## 4. Deep-math per leg

- **DSL — graduated non-convexity / homotopy.** The curriculum is a homotopy of relaxations of
  ONE energy functional S: CE (coarse partition) → tau_softplus (temperature/persistence anneal)
  → l7 (finest-scale) → Muon (Stiefel-orthogonalized escape). Stage transitions are first-class
  `ReTreat(rewarmup, reset_moments)` operators (different stages need different treatment). The
  gauge vocabulary (`gauge.py`) types the facets (Warp/Carrier/Residual/Pose/Movables/Generation/
  Head/Topology…) so a program is a checkable object, flag-validated against the real argparse.

- **equations — the confirmed laws.** e.g. margin field = Fisher surrogate (Pearson 0.978,
  the two are logit-derived → collinear by construction); UNIWARD = the same *detector-cost idea*
  but keyed to a GENERIC steganalyzer, so kindred-NOT-equal (pixelwise margin↔UNIWARD ≈ 0, measured
  n6 — the unity is metric-level via the Fisher/Jacobian, not a scalar-map correlation);
  saddle-to-saddle escape time ∝ exp(leap exponent) with Muon ≈ Stiefel
  (Spectral-Flattening 2605.13079); argmax-of-SDF ≡ power diagram (per-class offset = Laguerre
  weight); base-fiber ≡ task-space quotient (rate on base, uniform on fiber); the 96↔600
  subsample-bias law. Each carries Provenance (axis_tag + hardware + evidence_grade) + ≥1
  empirical anchor; `FORMALIZATION_PENDING` until measured.

- **DAG — the trajectory as data.** The FEED blocks are the measured history that the DSL's
  `decide` and the equations' recalibration both consume; the ordering (which scale locked in
  when) is itself the persistence spectrum the curriculum is trying to invert.

## 4b. The campaign-meta layer — the triality IS a POWERPLAY search (Schmidhuber, arXiv:1112.5309)

The double unification (§2) has a third, sharper reading: the *campaign itself* is a **POWERPLAY
search**, and this is now an equation (`powerplay_variant_ii_cost_isomorphism_v1`), not just prose.

- **S IS a POWERPLAY Variant-II cost.** `C(s)=L(s)+α·Σ_T[t'_s(T)−r(T)]` maps term-for-term onto
  `S = 100·d_seg + √(10·d_pose) + 25·|archive|/N`: `L(s)` (solver **description bits** / SPACE) = the
  rate term; `α·Σ[t'−r]` (weighted **task deficit** over the 2-task repertoire {SegNet-argmax,
  PoseNet-6}) = the distortion terms; `α` = the RD-Lagrangian λ. The identity
  `tac.witness_dsl.powerplay.powerplay_cost(x).S == tac.contest_score.compute_contest_score(x)` is
  **exact for all inputs** (VERIFIED_VIA_SOURCE_INSPECTION, residual 0.0).
- **The Correctness Demonstration = review axis-9.** POWERPLAY never ACCEPTS a solver-modification
  until a Correctness Demonstration proves (i) the new task is solved, (ii) no prior task regressed,
  (iii) the predecessor did not already solve it. Our launch-SEAL **axis-9** — *a SEAL is INVALID until
  it EXECUTES the real config and measures every scored quantity THROUGH the real byte-closed decode
  (never a proxy / ancestor / MPS / training-side surrogate)* — IS that Demonstration. The #205 SEAL
  failure (accepting a config on a borrowed ancestor d_pose with no runnability check → OOM) was
  accepting a modification on an *unproven* Demonstration. Executable:
  `tac.witness_dsl.powerplay.CorrectnessDemonstration`.
- **Variant-II acceptance `c*_pred − c_new > ε`** = the compose-without-regression / admit-only-when-
  net-S-improves gate. **`K(T,q | history)` simplest-still-unsolvable** = the principled `next()` for
  the #216 instrument (rank levers by ΔS per description+validation bit). Executable:
  `variant_ii_accept` / `simplest_unsolvable_rank`.
- **The two POWERPLAY cautions are OUR failure modes:** trivial-task-invention = the means-as-ends trap
  (levers that don't move the EXACT n600 S); generalization-vs-novelty tension = a real #211 corpus-
  generalize caution.

This is why Schmidhuber holds a grand-council seat: **task-aware compression = intelligence =
creativity-as-compression-progress** is the backbone, and POWERPLAY is its algorithmic skeleton. NOT a
contest lever (no through-R ΔS) — a campaign-structure law. Ledger:
`.omx/research/powerplay_1112.5309_deep_crossref_20260702.md`.

## 4c. Compression-as-intelligence grounding (why the task-space direction is PROVEN)

The equations leg carries the framing theorems that name and justify the machine (ledger
`.omx/research/compression_as_intelligence_lineage_crossref_20260702.md`):

- **Task R(D) < reconstruction R(D) — a THEOREM** (`task_rd_dominates_reconstruction_rd_v1`;
  arXiv:2602.12866 "Model-Aware Rate-Distortion Limits"; Dobrushin–Witsenhausen remote/indirect RD).
  For a fixed downstream model M, `R_M(D) ≤ R_X(D)` with gap = the task-irrelevant RGB-slack. So the
  NON-RGB task-space witness dominating a full-RGB codec is **proven, not asserted** — keeping any bit
  the frozen scorer never reads is provable rate waste. Kills any "just build a good RGB codec" revival.
- **The names for the machine we already built:** MDL-S (S is a two-part code); the **structure-function
  model/noise split** (Vereshchagin–Vitányi: `K(S)+log|S|=K(x)+O(1)` at the minimal sufficient
  statistic) = compile-the-generator (free deterministic generator = the model S; counted video-derived
  residual = the incompressible index); the **Speed-Prior / Levin-Kt** reading of the 30-min budget
  (minimize length + log runtime → the *fastest sufficient* generator wins, not merely the shortest);
  Sutskever's conditional `K(Y|X)` = the amortized meta-init / warm-start (#211). Naming = non-forgetting
  = the triality staying consistent.

## 4d. The layer ABOVE the triality — the Hamiltonian/optimal-control costate-controller (the canonical-consumer brain)

The triality has a **dimension above it, from which it falls out** (operator insight 2026-07-03, from the
EdgeBench arXiv:2512-class symposium; memory `meta-layer-above-triality-hamiltonian-optimal-control-costate`).
The three legs are not a free choice — they are the three unavoidable **shadows of one controlled learning
dynamics**: DAG = the **state trajectory** x(t), DSL = the **control policy** u(t), equations = the
**generator / action-law** S. That object is the learning process as a **Hamiltonian / optimal-control action**
(`δS/δθ = 0`, the "unified Lagrangian action" CLAUDE.md already half-named). In the D4 picture: the triality is
the **three outer nodes** (8v/8s/8c); the layer above is the **one central node** they hang off — the generator.
The DAG→DSL→equations cycle (§3) is the S₃/order-3 symmetry that rotates the shadows.

- **The missing 4th object — the COSTATE λ.** A pure {state, control, law} triality omits the adjoint: the
  **sensitivity of goal-value to state = the measured marginal-ΔS per lever** (grounded in the margin-saliency
  field #141 + the measured n600 rows). The costate is the shadow-price that flows DAG→DSL; **without it the DSL
  can only EMIT recipes, not OPTIMALLY SELECT** — a passive emitter, not the active controller the layer demands.
- **The measured equation of motion (EdgeBench).** Frontier expansion on a latent task graph gives
  `dx/du = β·x(1−x)` (logistic; the reaction term of Fisher–KPP), with `u ~ log t` from self-similar/fractal
  structure (they cite self-organized criticality). It is the SAME front-propagation equation at the **witness**
  scale (the level-set boundary, Layer A) AND the **campaign** scale (this triality, Layer B) — the meta-layer is
  scale-free, so the {state, control, law} triality precipitates at every zoom. HONEST caveat (fresh-eyes,
  App D.5): the clean log-sigmoid holds only on well-mixed graphs; a **bottleneck** (our d_seg ~8-dim lane
  manifold) yields plateaus / a sum-of-sigmoids — so the controller ranks by MEASURED marginal-ΔS per cost,
  NEVER by fitting the aggregate curve to a single bottlenecked run. FRAMING/VALIDATION, not a contest lever
  (no through-R ΔS). Ledgers: `.omx/research/edgebench_scaling_laws_deepdive_*.md`,
  `.omx/research/edgebench_freshpass_dynamics_derivation_*.md`, `.omx/research/gaussian_quant_2512.06609_deepdive_*.md`.
- **The form = the canonical-consumer bidirectional brain (operator: "sensors + actuators + turbo brain in one").**
  Close the DSL leg as an interpretable, ADVISORY (CONTAINMENT — never auto-fires paid/heavy GPU),
  continually-updated controller: `next_lever = argmax over READY levers of [predicted ΔS-toward-target · effect −
  cost]`, never-regress — simultaneously Pontryagin control-max, POWERPLAY frontier-selection (§4b), and
  learning-progress acquisition. **Sensors** = the `*_sensitivity*` producer family (the costate). **Brain** =
  `pareto_polytope_unified_solver`/`dykstra_pareto_solver`, `autopilot_rudin_daubechies/rashomon_ensemble`,
  `continual_learning`/`council_continual_learning`, `canonical_equations`. **Actuators** = `atom/atom.py`,
  `nerv_master_consumer_bridge`, the rate-allocator queue.
- **This is a DE-ORPHANING UNIFICATION, not a blank build.** We already have this brain sprawled from an earlier,
  clumsy attempt — the **cathedral autopilot** (`tools/cathedral_autopilot*.py` + `src/tac/{cathedral,
  cathedral_autopilot,cathedral_consumers,cathedral_solver_wire_in}`). It sprawled and orphaned producers because
  it was a *framework without a variational skeleton*; the Hamiltonian form is that skeleton (Carmack: a scoring
  function, not a framework). **The orphan problem restated = "a costate producer with no controller consuming
  it."** Build = task #247; Catalog #335 auto-discovery + the 6-hook wire-in are the standing anti-orphan enforcement.
- **The velocity-orphaning meta-bug (the deepest signal loss).** Even WITH that architecture, valuable things
  drifted / stagnated / were forgotten "due to extreme, sustained grueling dev velocity" — the producer-rate
  outran the consumption-rate, and consumption was a *manual step a relentless pace skipped*. The cure must survive
  the velocity that broke the last one: consumption **automatic + continuous**, the costate ranking its own
  orphans; re-consumption is **measured + costate-ranked, not a frantic sweep** (do not fight drift with more
  grind). Memory: `velocity-driven-orphaning-the-deepest-signal-loss-meta-bug`.

## 5. Consistency = campaign non-forgetfulness (the maintenance loop)

Per lever, ONLY after build + wire + integrate + rigorous test + recursive-adversarial-review
(3-clean-pass), update all three legs so they stay consistent (task #219, memory
`asbuilt-maintenance-loop-and-adversarial-experiment-gate`):

1. **equations** — `tac.canonical_equations.registry.register_canonical_equation` (Provenance +
   ≥1 `EmpiricalAnchor` + producers/consumers).
2. **DSL** — new `gauge` enum(s) + `COMPONENT_GAUGES`/`GaugeCost` cells; flag-validated vs the
   trainer argparse.
3. **DAG** — append a FEED block.

Plus the operator-facing surfaces (novel-contributions doc, writeup, θ* candidate configs). No
leg asserts a WIN — they record MEANS + advisory rows + honest negatives; a score claim requires
the byte-closed #202 exact row.

## 6. Adversarial + deep-math gates (the quality bar, applied to the triality)

Every automated cycle self-reviews before any verdict is load-bearing: n600-authority flag,
bit-exact-gate status, **automatic full-launch-line config-diff** (the discipline that would have
caught the n200/mod-32-vs-n600/mod-26 confounder *structurally*), MPS-never, confidence +
falsifier, 3-clean-pass recursive adversarial review. Negatives are SUSPECT — implementation-
level, not paradigm-level — until an adversarial + deep-math + OSS pass tries to overturn them.

## 7. Live campaign state (2026-07-03, the current thread)

- **Pointer 0.19110 UNMOVED** — the borrowed PR101/PR110 recode; the honest state until a byte-closed
  n600 `upstream/evaluate.py` row crosses it.
- **Machine CRASHED 2026-07-02 (concurrent jobs summed >128 GB → memory-pressure hang) → RECOVERED.**
  Root cause = per-process caps blind to the system total. Fix landed: the **system-memory governor**
  (`tools/system_memory_governor.py` + `memory_blackbox.py`) — a black-box recorder + system-wide
  admission HARD-gate (ENFORCING, `56147e797`; independently reviewed SAFE-TO-ENFORCE) + reversible
  throttle; canonical eq `adaptive_ceiling_admission_control_v1`. One lost-signal casualty (#238
  byte-close) re-tasked; recovery sweep `.omx/research/crash_signal_loss_recovery_sweep_20260703.md`.
- **R1 VERDICT (2026-07-03): store-nothing pose VIABLE, not yet optimal.** d_pose descended
  62.44→**0.0011** (plateau, ep1074/1093 ~0.00108); d_seg HELD ~0.0046 (no coupling collapse; seg⊥pose
  held). Viability CONFIRMED (the operator HOLD condition) — but contribution √(10·0.0011)=**0.105** is
  ~6× the ancestor 0.018 (~70% of a sub-0.15 budget). ⇒ the **POSE-CARRIER LADDER** (#248, DAG
  FEED-poseladder): P-A store-nothing (measured, 0.105) → **P-B FiLM 6-scalar stored-target** (target
  0.018 @ ~0.001 rate IF the witness render reads it back — the DECISIVE ~90:1-ΔS/byte rung) → P-C
  rank-residual interpolant / P-D warp-real fallback, run AS a rate↔pose **SWEEP** (map the Pareto,
  controller picks the knee). RATE-coupled (cheaper θ buys a better pose carrier). R1 stops at +1 point;
  byte-close #238 = P-A authoritative. d_seg is the remaining wall (lane/movable erasure → #205 islands).
- **#205 LAUNCH HELD** on operator GO, gated on R1's descent verdict + the two prerequisites (real optimal
  pose, confirmed byte-close). `--config sealed_205`/`store_nothing_205` reproduces the SEALed argv
  byte-identically. Pose carrier A/B ready (store-nothing-but-ξ vs table — rate collapse byte-close
  BIT-EXACT 1049 B / rate 0.0491). NO GPU till GO.
- **Meta-layer recognized (§4d):** the costate-controller / canonical-consumer brain = the de-orphaning
  unification of the cathedral autopilot + `*_sensitivity*` producers (task #247); its first act is the
  next-lever pick after R1's verdict + a costate-ranked re-consumption of velocity-drifted producers.

## 8. Pointers

- Concise awareness: CLAUDE.md § "The Triality — DAG ↔ DSL ↔ equations".
- Memories: `project-witness-dsl-and-dag-dsl-duality` (canonical), `unified-variational-levelset-flow-everything-is-facets` (Layer A), `all-automated-worldclass-recursive-review-deepmath-confirmed` (the automation bar), `asbuilt-maintenance-loop-and-adversarial-experiment-gate` (the sync loop).
- Code: `tac.witness_dsl.{campaign,curriculum_dsl,gauge,powerplay}`, `tac.canonical_equations`, `tools/levelset_byte_close_and_eval.py` (#202), `tools/witness_memory_preflight.py` (#205 OOM guard), `tools/dashboard_trajectory_model.py`.
- Campaign-meta + grounding ledgers: `.omx/research/powerplay_1112.5309_deep_crossref_20260702.md`, `.omx/research/compression_as_intelligence_lineage_crossref_20260702.md`.
- Equations (this session): `powerplay_variant_ii_cost_isomorphism_v1`, `oom_verdict_batch_spike_peak_rss_v1`, `task_rd_dominates_reconstruction_rd_v1`, `store_nothing_pose_carrier_rate_collapse_vs_dpose_v1` (all in `.omx/state/canonical_equations_registry.jsonl`).
- DAG: `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` (FEED-pp / -oom / -rdd / -219recon).
- Tasks: #216 (the automated instrument = the cycle engine), #219 (as-built triality sync), #189 (the DSL), #205 (the exact-row run it serves).
