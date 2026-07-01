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

- **equations — the confirmed laws.** e.g. margin field = Fisher surrogate = UNIWARD cost
  (Pearson 0.978); saddle-to-saddle escape time ∝ exp(leap exponent) with Muon ≈ Stiefel
  (Spectral-Flattening 2605.13079); argmax-of-SDF ≡ power diagram (per-class offset = Laguerre
  weight); base-fiber ≡ task-space quotient (rate on base, uniform on fiber); the 96↔600
  subsample-bias law. Each carries Provenance (axis_tag + hardware + evidence_grade) + ≥1
  empirical anchor; `FORMALIZATION_PENDING` until measured.

- **DAG — the trajectory as data.** The FEED blocks are the measured history that the DSL's
  `decide` and the equations' recalibration both consume; the ordering (which scale locked in
  when) is itself the persistence spectrum the curriculum is trying to invert.

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

## 7. Pointers

- Concise awareness: CLAUDE.md § "The Triality — DAG ↔ DSL ↔ equations".
- Memories: `project-witness-dsl-and-dag-dsl-duality` (canonical), `unified-variational-levelset-flow-everything-is-facets` (Layer A), `all-automated-worldclass-recursive-review-deepmath-confirmed` (the automation bar), `asbuilt-maintenance-loop-and-adversarial-experiment-gate` (the sync loop).
- Code: `tac.witness_dsl.{campaign,curriculum_dsl,gauge}`, `tac.canonical_equations`, `tools/levelset_byte_close_and_eval.py` (#202), `tools/dashboard_trajectory_model.py`.
- DAG: `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- Tasks: #216 (the automated instrument = the cycle engine), #219 (as-built triality sync), #189 (the DSL), #205 (the exact-row run it serves).
