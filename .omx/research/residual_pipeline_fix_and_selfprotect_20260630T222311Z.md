# Residual-INR hybrid pipeline — review-blocker FIX + self-protect landing

- **UTC:** 2026-06-30T222311Z
- **Authority:** `[macOS-CPU advisory] NON-PROMOTABLE`. The frontier pointer is **UNMOVED 0.19110**;
  nothing here moves it. This is composition PLUMBING + structural extinction of bug CLASSES (a MEANS).
  The pointer moves only when the residual-INR GPU run lands + the 4-section archive byte-closes +
  `upstream/evaluate.py` (CPU+CUDA) returns below 0.19110.
- **Inputs (the findings):** `.omx/research/residual_inr_adversarial_review_r1_20260630T214025Z.md`
  (HIGH-1, HIGH-2, MED-1, MED-2, LOW-1, LOW-2) +
  `.omx/research/residual_inr_determinism_automation_generalizability_audit_20260630T213729Z.md`
  (G1, A1.1, A3.1, A3.2).
- **Discipline:** CLAUDE.md "Bugs must be permanently fixed AND self-protected against" — every finding
  → a FIX **and** a STRICT `check_<bug_class>` gate (the 2-landing pattern).

## What this gates
The binding sub-0.15 run: the residual-INR hybrid is the "small INR (bulk OUTSIDE the counted weights)
shrinks the rate" hypothesis. Before this landing, firing the AUTOMATED pipeline would NOT have tested
that hypothesis (it emitted the superseded `--structured-init` no-shrink command + the wrong npz schema
+ a `SystemExit` stub) and the composition mask could not reach ~63% of the residual (a hidden geometry
ceiling that would have read as an INR-capacity wall). Both are now closed.

## FIXES (each with its proof)

### B1 / HIGH-1 / G1 — pipeline incoherence (the fire-blocker)
- **phase_a** now builds + saves the residual **TRAINING BUNDLE** the trainer's `--residual-mode`
  actually consumes (`generate_bulk_render_and_labels` → `build_residual_training_bundle` →
  `save_residual_training_bundle`, distinct filename `residual_bundle.npz`) and emits
  `build_residual_only_command` (`--residual-mode` + `--residual-target-npz`, **NOT** `--structured-init`).
- **phase_b** assembles the REAL 4-section archive from trained weights:
  `residual_blob_from_weights_npz` (reconstructs the forward cfg from the EMA-shadow npz `__cfg_*`/
  `__bank_*` scalars; mod_dim from the code shape; n_classes from `out_sdf`) → `build_residual_blob` →
  `pack_v2_archive`. The `SystemExit("NEEDS-WIRING")` stub is gone.
- **PROOF (inflate == train end-to-end, through the TOOL):**
  `test_compose_witness_archive_pipeline.py::test_phase_b_residual_archive_inflate_equals_oracle_end_to_end`
  builds the archive via the tool's `residual_blob_from_weights_npz` and asserts the subprocess
  `inflate.py` output is **bit-identical** (`np.array_equal`) to `residual_inflate_reference` (the numpy
  oracle). The bundle-schema handoff is proven by `test_bundle_schema_matches_trainer_consumer`
  (`load_residual_training_bundle` loads the phase_a bundle with no KeyError). Real-data smoke: phase_a
  on `gt_n96.npz` produces the bundle + emits the corrected command; phase_b byte-closes the floor
  (4112 B, rate 0.00011).

### B2 / HIGH-2 — composition mask must COVER the residual (geometry-ceiling gate)
- `derive_composition_mask` default mode is now **`boundary_annulus`**: the inter-class boundary of the
  bulk's OWN warped partition (dilated ~2px) — GT-free, **self-detected** (keys off where the partition
  changes; no SegNet class index hardcoded), covering ALL codim-1 flips (Road↔Undrivable/sky included),
  not just the Lane+Movable subset. `learn_classes` (legacy) + `union` are alternatives; the mode is
  carried into the bundle + the residual manifest so the inflate re-derives the SAME mask (train ==
  inflate).
- New `measure_composition_coverage(residual_mask, composition_mask, dseg_budget)` → coverage +
  `unreachable_dseg` (the d_seg the INR can NEVER close, a hard lower bound) + `passes_gate`. phase_a
  GATES the GO on `unreachable_dseg < sub-0.15 d_seg budget` and prints a NO-GO warning otherwise.
- **PROOF (the gate is load-bearing):** the n96 smoke fired correctly — coverage **0.1449**,
  `unreachable_dseg` **0.13194** ≫ budget **0.00123** → **FLAG (NO-GO)**. Without this gate that
  geometry ceiling would have been launched and misread as an INR-capacity wall. Unit tests:
  `test_measure_composition_coverage_gate` + annulus default/self-detection/union/bad-mode tests.

### B3 / MED-1 — pose axis honesty (pose = stored screw/twist, validate the read-back)
- Reframed d_pose as an **OPEN axis** in both phase reports. The pose sidecar stores the screw/twist
  (dual-use: warp→d_seg AND pose→d_pose); the `3.4e-5` figure is the stored-target **BUDGET**, NOT a
  measured composed d_pose. The overstated "pose solved / MEASURED 0.9KB" narrative is removed.
- phase_b adds an **advisory composed-d_pose measurement** (`--measure-dpose`): PoseNet(composed pair)
  vs the stored GT poses (which ARE PoseNet(original pair)) on a capped set of pairs (frozen CPU
  PoseNet, NEVER MPS). The s_budget uses the MEASURED d_pose when present, else the BUDGET clearly
  labeled. For the floor archive (f0==f1) PoseNet sees ~no motion → high d_pose; the residual INR
  (which differs f0/f1 in the override region) is what would close it. NO-FAKE #8: byte-measured !=
  d_pose-validated. (Per coordinator refinement: the fix keeps the pipeline amenable to ground-frame
  canonicalization — the stored twist is not assumed image-frame; the bigger structural refinements
  remain a separate follow-on design step.)

### Cheap closures
- **A1.1 (provenance):** `_provenance()` stamps `git rev-parse HEAD` + `compute_upstream_snapshot_sha256`
  + utc into BOTH phase reports (deterministic-reproducibility item 6).
- **A3.2 / LOW-1 (warp mask no longer dead):** the inflate's `_composite_warped` now **CONSUMES** the
  stored per-class `warp_codes` (derives the ground/rotonly/identity class lists from them) instead of
  the hardcoded `[0,1,3]/2/4` routing. The stored codes are the PHYSICAL `SCREW_REGIME` codes via
  `screw_regime_warp_codes` (`[0,0,2,0,1]` for the comma rig) — bit-identical to the proven
  `composite_warped_labels` router for the canonical clip (parity test green), generalizing on a clip
  whose regime differs.
- **A3.1 (one source of truth):** phase_a + phase_b both derive `warp_codes` via `_warp_codes_for_clip`
  (was: phase_b hardcoded `[0,3,2,3,1]`).
- **MED-2 (verdict↔inflate parity):** the trainer's verdict numpy forward (`_fwd_numpy`) calls the SAME
  `levelset_rgb_forward_numpy` + `where(mask,INR,bulk)` compose as the oracle, so the e2e inflate==oracle
  test covers verdict==inflate transitively.
- **LOW-2 (runtime closure):** the inflate's non-stdlib deps are numpy + torch (bicubic R) + brotli
  (residual section) — flagged for the byte-close packet smoke (pre-existing for the witness).

## SELF-PROTECT (the 2nd landing — 3 STRICT gates, all strict-flipped at live count 0)
- **Catalog #393 `check_orchestrator_emits_valid_trainer_contract`** (B1 + the META-bug "two correct
  halves, broken seam"): the residual launch emitters must flag-validate against the real trainer
  argparse (dynamic), AND the compose orchestrator must have an end-to-end inflate==oracle handoff test,
  assemble the residual archive (no stub), and emit the residual-only command (not the superseded one).
- **Catalog #394 `check_residual_override_has_coverage_proof`** (B2): the default mask mode must be the
  boundary annulus, and any override-builder must `measure_composition_coverage` + GATE on it.
- **Catalog #395 `check_axis_solved_claim_has_pipeline_validation`** (B3 / NO-FAKE #8): a "solved" axis
  claim at the v2 surface must carry a pipeline-validation qualifier (OPEN/advisory/measured-this-
  pipeline), not a borrowed claim.

All 3 wired into `preflight_all(strict=True)` (same-batch strict-flip, live count 0). Catalog rows in
`docs/meta_bug_class_catalog.md`. Folded the META-bug "require an end-to-end contract test" into #393
(gate-consolidation discipline #299, staying well under the #400 quota).

## Sister-substrate sweep
- **Override-mask bug class:** ONLY `tools/compose_witness_archive.py` builds a residual override (now
  fixed + gated). `bulk_generator` only GENERATES labels (no override). No sister.
- **Borrowed-axis-solved overstatement:** NONE remaining in the v2 compose CODE surface (the original
  "Pose is SOLVED" lived in the design memo narrative, not the tool). Gate #395 protects re-introduction.
- **Dead-flag / orchestrator-contract:** the other witness launchers (`launch_witness_run.py`,
  `levelset_byte_close_and_eval.py`, ...) already flag-validate (audit PASS evidence). #393's dynamic
  check covers the canonical residual emitters; no sister fix needed.

## Tests (all green)
- 60 v2_compose tests (incl. the bit-exact inflate-residual-compose parity with the annulus mask +
  consumed warp_codes).
- 6 `test_compose_witness_archive_pipeline.py` (B1 e2e contract + residual_blob_from_weights + bundle
  schema + emitter + physical warp codes).
- 28 `test_preflight_residual_pipeline_gates.py` (the 3 gates: positive/negative/waiver/edge/strict).
- Real-data smoke: phase_a (96 pairs, SegNet selfcheck OK) + phase_b floor byte-close.

## 6-hook wire-in (CLAUDE.md "no orphaned signals")
1. sensitivity-map — N/A (plumbing + gates; no per-axis byte-saving model).
2. Pareto constraint — N/A (no new rate/distortion knob; the coverage gate is a feasibility bound, not
   a Pareto term).
3. bit-allocator — N/A.
4. cathedral autopilot dispatch — the corrected `build_residual_only_command` is the dispatch the
   campaign would fire; the coverage gate is a pre-fire feasibility check the autopilot can consult.
5. continual-learning posterior — N/A this landing (advisory; the byte-closed exact row is the anchor).
6. probe-disambiguator — the coverage gate IS the disambiguator between "geometry ceiling" and "INR
   capacity wall" (the false-capacity-wall risk HIGH-2 named).

## Bottom line
The residual pipeline is now **coherent end-to-end** (inflate == train == oracle proven through the
tool), the geometry ceiling is an **explicit pre-fire gate** (fired NO-GO on the n96 smoke), the pose
axis is **honestly OPEN** (advisory measurement, no overstatement), and the four bug CLASSES (broken
seam / geometry-ceiling override / borrowed-solved-claim / the META-bug) are **structurally extinct**
via 3 STRICT gates. **READY for R2 re-review.** Pointer UNMOVED 0.19110 (a MEANS, per NO-FAKE means≠ends).
