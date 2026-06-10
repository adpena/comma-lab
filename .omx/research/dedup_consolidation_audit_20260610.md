# Dedup / Consolidation Audit — Evaluator-Inverse Tool Wave — 2026-06-10

**Subagent:** `dedup_consolidation_audit_20260610`
**Operator concern (2026-06-10):** the last two sessions built many evaluator-inverse
tools fast; some may duplicate preexisting surfaces — *"we don't want to reinvent
the wheel or clutter the codebase with duplicative code."*
**Baseline:** `.omx/research/evaluator_inverse_orphan_inventory_20260609.md` (the 103-surface map).
**Evidence grade:** `[macOS-CPU advisory]` / structural audit. No score claims; no dispatch.
**Scope guard:** did NOT touch files owned by RUNNING sister agents (snerv_g1b /
snerv_branch_b run dirs, the frontier_latent_axis latent surfaces, pr101 parsers,
latent blob tools, `modal_auth_eval_cpu.py`). Verified via
`.omx/state/subagent_progress.jsonl` — active sisters touch only SSD run dirs +
their own research memos; zero overlap with the audited optimization modules.

---

## Headline result

| Metric | Count |
|---|---|
| New modules audited | 13 modules + 7 tools + 4 `_shared` manifests |
| **TRUE-DUPLICATE found** | **1** (`_measure_exact_distortion` triple) |
| **TRUE-DUPLICATE consolidated** | **1** (delegated to canonical home) |
| **COMPLEMENT documented** | **4** (invisibility-basis↔xray, cone↔z8, pose-atom resize, atoms↔cone) |
| **LAYERED verified** | **9** (all 4 manifest data↔engine pairs, 5 tools↔modules, lf_payload↔atlas↔cone) |
| **Documentation-fidelity fix** | **1** (cone docstring claimed a z8 reuse it did not perform) |
| **Systemic pre-existing finding (NOT this wave's fault; documented, not mass-rewritten)** | contest-constant proliferation (`37_545_489` named ~10 ways across ~20 modules) |
| LOC removed (net duplicate logic) | ~24 LOC of duplicated function bodies (3×8) replaced by 3 one-line aliases + 1 canonical promotion |

**The single worst duplication found:** `src/tac/optimization/frame1_joint_safe_cone.py`'s
module docstring (lines 37-49) **claimed** it *"reuses, not rebuilds"* z8's
`segnet_boundary_pixel_saliency` and `posenet_pixel_jacobian_norm`, but the code
imports NO `tac.` module and re-implements those scorer forwards inline as
`measure_segnet_frame1_margin` / `measure_posenet_frame1_jacobian`. This is the
NO-FAKE class (a reuse claim the code does not honor). It is NOT a true-duplicate
to delete (the cone genuinely needs frame1-specialized OUTPUTS z8 does not emit —
see §C), but the misleading reuse claim was corrected so the docstring matches the
code. The closely-related `_measure_exact_distortion` triple (§B) was the only
literal byte-for-byte duplicate and was consolidated.

---

## A. Full mapping table — NEW surface → what it computes → OLD surface → verdict

Verdict legend: **TRUE-DUP** (same math+output, consolidate) · **COMPLEMENT**
(different math/domain/output — boundary documented in both) · **LAYERED** (new
correctly consumes old; import verified) · **CLEAN-NEW** (no existing surface).

| NEW surface | Computes (file:line) | Closest OLD surface | Verdict |
|---|---|---|---|
| `optimization/frame1_joint_safe_cone.py` `measure_segnet_frame1_margin` (l.225) / `measure_posenet_frame1_jacobian` (l.289) | raw SegNet top-2 **margin + argmax class** (frame1); frame1-channel-only PoseNet Jacobian `(H,W)` + fail-closed zero guard | z8 `joint_p18_p19_deadzone_rate_attack.segnet_boundary_pixel_saliency` / `posenet_pixel_jacobian_norm` | **COMPLEMENT** — same scorer forward, DIFFERENT output (z8 emits `exp(-margin/τ)` saliency + `(2,H,W)` both-frame norm on the coeff grid; cone needs raw margin+argmax + frame1-only norm + zero-guard). Docstring corrected (was a false reuse claim). |
| `frame1_joint_safe_cone.py` `_measure_distortion` (l.541) | `(d_seg,d_pose)` via `dn.compute_distortion` | (canonical home) | **CANONICAL** — promoted to public `measure_pair_distortion`; the home for the §B triple. |
| `optimization/frame1_seg_safe_pose_atoms.py` `_measure_exact_distortion` (was l.505) | identical `dn.compute_distortion` wrapper | cone `measure_pair_distortion` | **TRUE-DUP → consolidated** (now `= measure_pair_distortion`). |
| `optimization/frame1_seg_repair_atoms.py` `_measure_exact_distortion` (was l.573) | identical `dn.compute_distortion` wrapper | cone `measure_pair_distortion` | **TRUE-DUP → consolidated** (now `= measure_pair_distortion`). |
| `frame1_seg_safe_pose_atoms.py` `_resize_map` (l.370) | **numpy-portable** separable bilinear (torch-free, Catalog #383) | cone `_resize_to_grid` (l.348, torch) / z8 `_resize_pixel_map_to_grid` (torch) | **COMPLEMENT** — deliberately torch-free for the numpy-portability contract; correctly reused by repair via import. Keep distinct. |
| `frame1_seg_safe_pose_atoms.py` / `frame1_seg_repair_atoms.py` (the atom generators) | Class-2 seg-safe pose atoms / Class-3 seg-repair atoms; consume cone `ConeFields.from_cone` | (no prior frame1-atom surface) | **CLEAN-NEW + LAYERED** (consume the cone; repair imports pose `_resize_map`). |
| `optimization/evaluator_response_atlas.py` (l.247 `build_atlas_row_from_cone`, l.402/462 cross-video reduce) | per-pair scorer-field SUMMARY rows + MLX/numpy cross-video reduce; **no scorer forward of its own** | `analysis/action_effect` IR (referenced) + `cathedral_consumers/per_pair_difficulty_atlas_consumer` | **CLEAN-NEW (the index/engine) + LAYERED** — consumes cone output as input; the row/law/vocabulary it references are the existing IR (docstring-referenced, not re-authored). |
| `optimization/evaluator_invisibility_basis.py` (l.116 `_resize_1d_matrix`, l.310 tier-1 null) | **closed-form EXACT** separable bilinear resize null space (camera-pixel domain; residual==0.0 certified) | `xray/bilinear_resize_nullspace.py` (Monte-Carlo Hutchinson **estimator**) + `null_space_exploiter` (**byte-space**) | **COMPLEMENT** — estimator vs certification-grade derivation; pixel-domain vs byte-domain. Already cross-referenced in both directions (l.20-21, l.67, l.70). Keep distinct. |
| `optimization/resize_null_preimage.py` (l.310+) | minimum-description PREIMAGE postprocessor `argmin bytes(x̃) s.t. R x̃ = R x` | (no prior) — imports invisibility_basis (l.101) + lf_payload `delta_rate_score` (l.726) | **CLEAN-NEW + LAYERED** (real imports; does not rederive R). |
| `optimization/lf_payload_rate_distortion.py` (l.830 `delta_distortion_score`, l.851 `keep_component`) | evaluator-conditioned reverse-waterfill THE LAW | (orphan inventory: this IS task #46; single clean module) | **CLEAN-NEW (canonical #46)** — created once, iterated in place; widely reused by atlas+tool+resize. |
| `analysis/scorer_spectral_sensitivity_v2.py` | transfer-function spectral atlas (v2) | (no live v1 module; v1 was the analyzer the v2 docstring supersedes) | **CLEAN-NEW (supersedes v1 by pointer in docstring)**. |
| `optimization/audit_provenance.py` | typed audit-claim records (mandatory surface + reproduce_command) | `_shared/constants_provenance_audit.py` (different: constants audit, not claim provenance) | **CLEAN-NEW** — no duplicate. |
| `_shared/vehicle_fidelity_manifest.py` / `objective_reachability_manifest.py` / `constants_provenance_manifest.py` | schema/engine: frozen dataclass + `verify()` + `emit` | their `*_manifests_canonical.py` siblings (the seed DATA) | **LAYERED** — canonical-data files import the engine (verified l.38/39/50). Engine↔data split is the endorsed pattern, NOT duplication. |
| 7 tools (`build_*`, `run_*`, `measure_*`, `resize_*`, `snerv_lf_*`, `render_*`) | thin CLIs | their modules | **LAYERED** — every tool imports its module; none re-author logic (verified). Dashboard delegates to `comma_lab.pact_compiler_dashboard` (correctly in comma_lab). |

---

## B. Executed consolidation — `_measure_exact_distortion` triple → `measure_pair_distortion`

**The duplicate:** three byte-for-byte identical 8-line functions:
- `frame1_joint_safe_cone._measure_distortion` (l.541)
- `frame1_seg_safe_pose_atoms._measure_exact_distortion` (l.505)
- `frame1_seg_repair_atoms._measure_exact_distortion` (l.573)

Each body was exactly:
```python
import torch
with torch.inference_mode():
    d_pose, d_seg = dn.compute_distortion(gt_pair.float(), cand_pair.float())
return float(d_seg.mean()), float(d_pose.mean())
```

**The consolidation (delegation shim + deprecation by alias; APPEND-ONLY — no
history deleted):**
1. Promoted the cone's helper to a public canonical `measure_pair_distortion`
   (the #35 cone is the foundational surface both atom classes already consume),
   added it to `__all__`, kept `_measure_distortion = measure_pair_distortion`
   for the cone's own 3 callsites.
2. `frame1_seg_safe_pose_atoms`: added top-level
   `from tac.optimization.frame1_joint_safe_cone import measure_pair_distortion`;
   replaced the local def with `_measure_exact_distortion = measure_pair_distortion`
   (alias preserves the 2 in-module callsites verbatim).
3. `frame1_seg_repair_atoms`: added the same import (next to the existing
   `_resize_map` import from pose-atoms); replaced the local def with the alias.

**Verification:**
- `test_frame1_joint_safe_cone.py` + `test_frame1_class23_atoms.py` +
  `test_evaluator_response_atlas.py` = **77 passed** (was 131 incl. basis/preimage
  pre-edit; the 77 cover every edited module).
- Identity smoke: all three module aliases `is measure_pair_distortion` (True).
- `ruff check` clean on all 3 edited files.

**Net:** ~24 LOC of duplicated function bodies → one canonical 9-line function +
3 one-line aliases. Single source of truth for the exact CPU-torch pair-distortion
measurement; future changes (e.g. an MPS-guard or a batched variant) land once.

---

## C. Keep-distinct rationales (the COMPLEMENTs — why duplication-looking surfaces are NOT duplicates)

1. **cone scorer-forwards vs z8 `*_saliency`/`*_jacobian_norm`.** Same SegNet/PoseNet
   forward, DIFFERENT output by design: z8 emits flip-proneness *saliency*
   (`exp(-margin/τ)`) and a both-frame `(2,H,W)` Jacobian on the **wavelet-coeff
   grid** (it feeds the z8 detail-coeff dead-zone). The cone needs the **raw margin
   + argmax class** (distance-to-flip budget + behavioral-check reference) and the
   **frame1-channel-only** `(H,W)` Jacobian + a fail-closed zero-energy guard.
   Wrapping z8 would require post-processing its output back into the cone's form
   AND z8 is a substrate-private module; deriving directly is the correct per-method
   engineering (UNIQUE-AND-COMPLETE-PER-METHOD). **Action taken:** corrected the
   cone docstring (it falsely said "reused, not rebuilt") to state the true math
   lineage (sister forward) without claiming a code reuse that does not occur.

2. **`evaluator_invisibility_basis` (#47) vs `xray/bilinear_resize_nullspace` vs
   `null_space_exploiter`.** Three different things: (a) NEW = closed-form EXACT
   resize-preprocess null in the **camera-pixel** domain (residual==0.0,
   hardware-independent certification); (b) OLD xray = Monte-Carlo Hutchinson
   **estimator** of the same null fraction (approximate, the NEW module certifies
   what xray estimates); (c) OLD `null_space_exploiter` = **byte-space** null basis
   over the master-gradient `(n_bytes,n_pairs,n_axes)` tensor, wired into
   `unified_action`. Different domains + different math. Already cross-referenced
   bidirectionally in the new module's docstring. No action needed.

3. **`frame1_seg_safe_pose_atoms._resize_map` (numpy-portable) vs cone/z8 torch
   resize.** Deliberately torch-free per the MLX-FIRST numpy-portability contract
   (Catalog #383). The atoms run their leverage search in numpy/MLX (no torch
   dependency) and only invoke torch for the exact CPU screening. Repair correctly
   imports it. Keep distinct (collapsing it onto the torch resize would re-introduce
   a torch dependency into the portable search path).

4. **manifest engine vs `*_manifests_canonical` data.** The schema/engine
   (`verify()`, frozen dataclass, `emit`) is correctly separated from the seed
   DATA (`emit_all()` of populated instances). The data files import the engine.
   This is the endorsed pattern, not duplication.

---

## D. Systemic pre-existing finding (DOCUMENTED, deliberately NOT mass-rewritten)

**Contest-constant proliferation.** The canonical contest coefficients live in
`tac.score_geometry` (`SEG_COEFFICIENT=100`, `POSE_COEFFICIENT_INSIDE_SQRT=10`,
`RATE_COEFFICIENT=25`, `CONTEST_REFERENCE_BYTES=37_545_489`; used by 33 production
modules). But the rate denominator `37_545_489` is **independently named ~10 ways
across ~20 modules** (`CONTEST_ORIGINAL_BYTES`, `ORIGINAL_VIDEO_BYTES`,
`RATE_DENOMINATOR_BYTES`, `CONTEST_RATE_DENOM_BYTES`, `CONTEST_ARCHIVE_RATE_DENOM`,
`FRAME_BYTES_DENOMINATOR`, `CONTEST_N_BYTES`, …), and the new wave's modules
(lf_payload, the 2 atom modules, the cone) re-declare local `_SEG_COEF`/`_RATE_COEF`/
`_POSE_TEN`/`_CONTEST_TOTAL_BYTES` numeric literals. **The values all match**, so
this is a drift HAZARD, not a live bug.

**Why NOT consolidated in this pass:** (a) it is a deep PRE-EXISTING repo
convention the new wave merely followed (the new modules did not introduce the
proliferation — they inherited it); (b) the ~20 re-declarations live across modules
owned/touched by other agents and the latent-axis surfaces this audit must not
touch; (c) a mass-rewrite of contest constants is exactly the kind of high-churn,
cross-agent edit that risks a body-shuffle / sister-conflict for marginal value.
The clean, low-risk fix is the prevention checklist in §F (new modules import the
canonical constants from `score_geometry` instead of re-declaring literals) — to be
applied to NEW modules going forward and to OLD modules opportunistically when
already being edited for another reason. This is flagged for the
`constants_provenance` discipline owner as a candidate canonical-helper landing
(a single `tac.contest_constants` re-export of the `score_geometry` four, that the
20 modules delegate to over a deprecation window).

---

## E. ORPHAN status of the new wave (none are orphaned)

Every new module has a non-test production importer (the thin CLI tool or a sister
module), per §A LAYERED column. None require the `research_only=true` orphan tag.
The wave is well-integrated: cone → atoms → atlas → lf_payload → resize-preimage
form a clean producer→consumer DAG, each tool wires its module, and the
invisibility-basis is consumed by the resize-preimage compiler.

---

## F. PREVENTION — pre-build checklist (extends the orphan-inventory REUSE PLAN)

Before authoring ANY new evaluator-inverse / scorer-exploit / sensitivity /
waterfill / null-space surface, a build subagent MUST:

1. **Grep the orphan inventory first.**
   `grep -i "<concept>" .omx/research/evaluator_inverse_orphan_inventory_20260609.md`
   — it maps 103 existing surfaces to tasks. If your concept appears under a task's
   REUSE PLAN, IMPORT the named surface; do not re-author it.

2. **Grep for the math, not the name.** A surface may exist under a different name.
   For scorer forwards: `grep -rln "compute_distortion\|posenet.*autograd\|topk(.*dim=1" src/tac`.
   For contest constants: **import from `tac.score_geometry`** (`SEG_COEFFICIENT`,
   `POSE_COEFFICIENT_INSIDE_SQRT`, `RATE_COEFFICIENT`, `CONTEST_REFERENCE_BYTES`) —
   NEVER re-declare `100.0`/`25.0`/`10.0`/`37_545_489` as a local literal (§D class).
   For pair-distortion measurement: import
   `tac.optimization.frame1_joint_safe_cone.measure_pair_distortion` (the canonical
   home as of 2026-06-10) — do not write another `dn.compute_distortion` wrapper.

3. **If your docstring says "reuses X", the code MUST `import X`.** A reuse claim
   without a corresponding `import` is the NO-FAKE class (the worst finding here:
   the cone claimed to reuse z8 functions it actually re-derived). Either import it,
   or state honestly "sister of X (same math, different output)".

4. **Different output ≠ duplicate; same output = duplicate.** If you genuinely need
   a different OUTPUT from the same forward (raw margin vs saliency; frame1-only vs
   both-frame), deriving directly is correct UNIQUE-AND-COMPLETE-PER-METHOD — but
   say so in the docstring (§C pattern). If the output is identical, import.

5. **Engine vs data split is fine; two engines is not.** A `*_manifest.py` (schema)
   + `*_manifests_canonical.py` (seed data importing the schema) is the endorsed
   pattern. Two modules each re-declaring the same dataclass is duplication.

6. **Tools are thin.** A new `tools/*.py` must import + wire existing modules, never
   re-author logic. The dashboard pattern (delegate to a `comma_lab.*` reusable
   module) is the model.

---

## G. Audit-provenance index (reproduce commands)

- New-module existence + LOC + last commit: `for f in ...; do wc -l $f; git log -1 --format='%ai' -- $f; done`
- Cross-ref scan: `grep -rlE "<old_surface>" src/tac/optimization/<new_module>.py`
- Triple-duplicate confirmation: `awk '/^def _measure_(exact_)?distortion/{p=1}...' <each module>`
- Contest-constant proliferation: `grep -rnE "= *37_545_489|= *37545489" src/tac --include='*.py' | grep -v test_`
- Consolidation green: `.venv/bin/python -m pytest -q src/tac/tests/test_frame1_joint_safe_cone.py src/tac/tests/test_frame1_class23_atoms.py src/tac/tests/test_evaluator_response_atlas.py`
- Alias identity: `python -c "from tac.optimization.frame1_joint_safe_cone import measure_pair_distortion; from tac.optimization import frame1_seg_safe_pose_atoms as p, frame1_seg_repair_atoms as r; assert p._measure_exact_distortion is measure_pair_distortion is r._measure_exact_distortion"`
- Score-geometry canonical constants: `grep -nE "^SEG_COEFFICIENT|^RATE_COEFFICIENT|^CONTEST_REFERENCE_BYTES" src/tac/score_geometry.py`
