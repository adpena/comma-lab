# θ*-residual-INR level-set config — UPDATE REQUIREMENTS (incorporate ALL)

**UTC** 2026-06-30T22:00Z · **tag** `[design-refine requirements · advisory · NON-PROMOTABLE]` · **pointer 0.19110 UNMOVED.**

Operator 2026-06-30: *"Our latest theta star residual INR level set config will need to be updated accordingly with all."* This is the anti-signal-loss capture of EVERY refinement the review gauntlet (3 passes) + the screw/twist research + the operator's pose=screw/canonicalize insights surfaced. The **design-refine step** synthesizes this into the actual updated config + the flag-validated launch command; **R2 re-review** to 3-clean; then **fire**. The inherited config (mod-16/hidden-48/epochs-1500/--structured-init) is SUPERSEDED. Every value below carries provenance: **MEASURED / DERIVED / SOLVED / LEARNED** (never hardcoded — the standing bar).

## What the updated config MUST incorporate

### Pipeline coherence (B1 / HIGH-1 / G1 — fire-blocker)
- `--residual-mode` consuming the residual **training bundle** (`save_residual_training_bundle`), NOT the superseded `--structured-init`; PHASE-B assembles the real 4-section archive (`build_residual_blob`); inflate composes bulk⊕residual. **End-to-end parity (inflate==train) is the gate.** [fix: a0e42df5]

### Composition mask = boundary ANNULUS (B2 / HIGH-2 — geometry-ceiling fix)
- Override mask = the GT-FREE inter-class boundary annulus (from the bulk's own argmax boundaries, dilated) — COVERS all flip mass (incl. Road↔Undrivable/sky ~63%), not just bulk-predicts-{Lane,Movable}. + the **$0 coverage measurement gate** (mask covers bulk≠GT ≥ threshold) before GO. **SOLVED** (annulus = the codim-1 residual). [fix: a0e42df5]

### Pose = the screw/twist, encoded ONCE (B3 + operator + deep-math Q6)
- Pose is the stored screw/twist ξ∈se(3) (dual-use: warps partition for d_seg AND is the pose for d_pose; OR "falls out of the level set" via the screw's temporal action). ~FREE (not a separate 0.9KB sidecar). **VALIDATE the read-back** (does PoseNet read ξ back off the witness; fallback = store the PoseNet-6-vector). **DERIVED/MEASURED.**

### Canonicalize residual to the GROUND FRAME (deep-math Q6 + pose=screw — the structural win)
- The residual INR operates in the canonical/ground frame with the ego-motion REMOVED (the screw encoded once). Residual code collapses to ~2-4 dims; the ~65KB image-space rate vanishes. The residual carries ONLY what does NOT fall out of the screw action. **SOLVED** (the deeper structural refinement).

### Architecture — DERIVE, don't inherit (deep-math fractal pass)
- **mod-dim:** run the **$0 residual-ID measurement** (TwoNN/MLE on `residual_target.npz`) FIRST → mod-dim = Whitney 2m+1 of the RESIDUAL sub-manifold (measured ID 8-13 → ~19-21 image-space; lower after ground-frame canonicalization). NOT the inherited mod-16 (under-embeds). **DERIVED/MEASURED.**
- **hidden-dim:** re-derive — the residual is ALL high-frequency boundary detail; hidden-48 (a 4× cut) is flagged risky. **DERIVED.**
- **Movables → STORE** (multibody codec ≤2.7KB → d_seg ~0.0008); the INR carries **lane-survival only**. **SOLVED.**

### Curriculum / epochs — re-derive for the residual (deep-math Q2)
- Skip/shorten CE (no smooth-bulk regime to warm up); early-stop at the residual knee (1500 is an inherited over-train guess); per-stage treatment for the rougher multi-modal landscape. **DERIVED.**

### Warp + screw-blend (research enrichment)
- Per-class SE(3) action via the **MLX se(3) lib** (exp/log/adjoint/J_r, numpy-parity-gated); **dual-quaternion screw-blend at the class boundaries** (the annulus seam — fixes the linear-blend tear exactly where d_seg is scored); ξ_ego(t) as a cumulative SE(3) B-spline (~24-48 floats). Metal-accelerate the pixel-grid warp/blend IF profiled hot. **SOLVED + DERIVED.** [research: a7eda614]

### Byte allocation (deep-math Q5)
- Residual code coding = temporal-AR + low-rank (nuclear-norm) aware, NOT brotli-blind; + the PR101 weight ladder. **DERIVED.**

### Determinism / automation / generalizability (audit AXIS-1/2/3)
- Provenance (git-hash + upstream-snapshot-sha) in every result/launch; ONE entry point (compose_witness_archive PHASE-A/B); clip-agnostic machinery (overfit in DATA not code; split self-detects by signature, never class index). **MEASURED.**

### Self-protect gates (operator: fix + protect bug classes + meta-bug)
- The STRICT gates from the fix: orchestrator-emits-valid-trainer-contract (B1 class), residual-override-coverage-proof (B2 class), axis-solved-claim-has-pipeline-validation (B3 class), the end-to-end-handoff-contract meta-gate. [fix: a0e42df5]

### Telemetry / provenance discipline (standing bar)
- Every config value tagged MEASURED/DERIVED/SOLVED/LEARNED; "calibrating" where not yet determinable; no hardcoded constant masquerading as truth.

## Mechanism
The design-refine step (after a0e42df5 fix + a7eda614 research + the $0 residual-ID measurement land) produces the actual updated config via the auto-config actuator (`witness_autoconfig` — derive-from-measured) + these refinements → the flag-validated `--residual-mode` launch command → R2 3-clean review → fire → byte-close → dual exact eval. means≠ends: this is the optimal-form CONFIG (a MEANS); the pointer moves only on the byte-closed exact row.

## Triality unification (DAG ↔ DSL ↔ equations) — keep ALL THREE updated + consistent
Operator 2026-06-30: *"keep ensuring the triality is unified and updated."* Per [[project-witness-dsl-and-dag-dsl-duality]], the updated config is ONE object in THREE views; the design-refine MUST propagate EVERY refinement above to all three, kept consistent (no drift) — and this is a STANDING check for every future refinement, not one-time:
- **EQUATIONS** (`tac.canonical_equations`): register/update — **pose-as-screw** (d_pose target = ego-motion twist ξ; PoseNet(pair)[:6] ≈ ξ ∈ se(3)); **canonicalize-to-ground-frame** (S_τ written in the ground frame; residual = GT − W(ξ_ego(t))∘Φ_canon through R); **dual-quaternion screw-blend** at class boundaries; **ξ_ego(t) = cumulative SE(3) B-spline**; **movables-stored** (out of the INR). Extend the E0–E8 master-action system; attach an EmpiricalAnchor to each when measured.
- **DSL** (`tac.witness_dsl`): extend the declarative program with the new constructs — `residual_mode(target)`, `ground_frame_canonicalize(screw)`, `per_class_warp{road=homography, sky=rotation_only, hood=identity, blend=dual_quaternion}`, `se3_bspline(controls)`, `store_movables(codec)`, `mod_dim=derived` — so the updated config IS a DSL program that compiles (flag-validated against the real argparse) to the `--residual-mode` launch command.
- **DAG** (`.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_*`): append the FEED entries for this thread — the gauntlet 3-pass findings (B1/B2/B3), pose=screw / "falls out of the level set", canonicalize-to-ground-frame, the MLX se(3) + Metal-Lie research, this config-update — so the trajectory view stays current.
- **Consistency invariant:** the DSL program compiles to the command the DAG records, governed by the equations — three views, one object. The design-refine verifies all three agree before R2/fire (a $0 check), and re-verifies on every future refinement.
