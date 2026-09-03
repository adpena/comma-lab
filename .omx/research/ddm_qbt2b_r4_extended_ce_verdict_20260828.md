# ddm_qbt2b r4 extended-CE verdict — flip 0.01573 at the 1,000-step cap (censored, still falling); Lane NEVER BIRTHS under UNWEIGHTED CE (pred 0.0051% vs GT 0.60%, within-err 99.76%); Movable also above gate (27.84%); cure = r5 balanced inverse-frequency CE (BUILT, sealed, matched control = r4)

STORES CONSULTED: r3 verdict memo (`ddm_qbt2b_r3_ce_birth_verdict_20260827.md`) · qbt2b charter
(`.omx/research/charters/ddm_qbt2b_inherited_palette_birth_20260828.md`) · #315/#686 derived-schedule
law · m131 Lane-demand law (Lane = 90.1% of the frontier rate demand) · m88/#1251 single-seed ·
m143 cross-regime transfer · #1313 R2-ENOSPC postmortem · retained r4 verdict payloads (re-read
from disk this session, not working memory, per m44).

score_claim=false everywhere. All rows [macOS-Metal/CPU frozen-scorer advisory, n32
seeded-stratified, single seed 20260827]. Pointer UNMOVED (gb1 0.14811799921260607).

## 1. The run (MEASURED)

- r4 = the r3 continuation: extended CE birth window 100 → 1,000 + unchanged margin+pose stage
  (5,000) on #315 event handoff. Fresh compile (config sha 4f5326a4c77b…), launch counter 689,
  pid 37053, rc=0 clean at the 1,000-step cap. Endpoint artifact
  `stage_03a_cap_without_handoff.pt`; 200 verdicts + 200 periodic checkpoints retained (payload
  law). Lanes ddm_qbt2b_r4_metal/scorer_20260827 closed terminal
  (`completed_cap_without_handoff_rc0`).

## 2. THE ENDPOINT (MEASURED, re-verified from `verdict_0200_step_001000.npz` this session)

Trajectory (realized flip, 32 pairs, 512×384 argmax vs target):

| step | 100 | 500 | 900 | 1,000 |
|---|---|---|---|---|
| flip | 0.06105 | 0.01962 | 0.01632 | **0.01573** |

Still falling at the cap — 0.01573 is a CENSORED bound. Pose co-descended: pose_mse 0.002248 at
step 1,000 (vs 128.3 at r3 step 5). Per-class at step 1,000:

| class | GT area % | predicted % | within-class err % | gate (<20%) |
|---|---|---|---|---|
| Road       | 23.12 | 23.88 | 1.28  | PASS |
| Lane       |  0.60 | 0.0051 | **99.76** | **FAIL — never predicted** |
| Undrivable | 49.58 | 49.68 | 0.49  | PASS |
| Movable    |  1.25 | 0.97  | **27.84** (falling: 77.43 @100) | **FAIL** |
| MyCar      | 25.46 | 25.46 | 0.36  | PASS |

The #315 event gate (all 5 classes within-err <0.20, 2 consecutive verdicts) therefore correctly
REFUSED the margin handoff — TWO classes failing, not one. The margin+pose stage never ran.

## 3. Adjudication (the #1314 fork, r4 half)

- Fork (b) from the r3 memo fires with a REFINEMENT: "Lane still unborn at 1,000 → the CE-alone
  Lane verdict hardens" — but hardening to FORMULATION scope now would violate the charter-time
  optimal-form law (#307 / naive-at-charter, m140). UNWEIGHTED per-pixel CE gives each class a
  gradient share equal to its area share: Lane at 0.60% area receives 0.6% of the gradient — a
  MECHANISM-reduction relative to the long-tail family's standard form (inverse-frequency /
  balanced weighting), not a scope-reduction. The honest verdict scope for "CE cannot birth
  Lane" is INSTANCE (unweighted CE, this init, 1,000 steps, n32, single seed) — NOT family.
- The pre-registered r3 falsifier is unchanged: CE births where the margin law froze (#315/#686
  law, 2 measured anchors). The r4 window sharpened WHERE unweighted CE saturates: bulk classes
  reach 0.36–1.28%, the two rare classes (0.60% + 1.25% area) are exactly the residue. This is
  m131's Lane-demand law showing up in TRAINING dynamics, not just rate accounting.

## 4. The cure (BUILT + SEALED this session — r5)

- `derive_balanced_class_weights`: w_c = total_pixels/(K·count_c) DERIVED AT RUNTIME from the
  sealed selection's real GT targets (no hand-typed constants, m47); weighted CE normalized by
  Σw[target] so loss magnitude matches r4; config-gated `birth_class_weight_mode` ∈ {"none",
  "balanced"}, default "none" = legacy byte-identical; mode participates in `config_identity`
  (cross-mode resume refused by design); weights stored in `curriculum_state` for provenance.
  Landed 7bc66c0da3 (trainer + 3 tests, 16/16) after 2-pass review.
- Balanced weights at this selection: Lane ≈ 33× and Movable ≈ 16× the mean-pixel weight —
  the two failing classes get exactly the gradient share the unweighted form denied them.
- r5 authorized config: `AUTHORIZED_N32_R5_6000_20260828.json` sha 307b4e48e810…, mode=balanced,
  seed 20260827 + SELECTION_IDS + init sha 0bedbd66… + schedule ALL IDENTICAL to r4 ⇒ **r4 is
  the exact matched control** (single-variable discipline, m52/m85). Lanes
  ddm_qbt2b_r5_metal/scorer_20260828 claimed. EMA lawref identical (0.9992327661102197).
- r5 forks: (a) Lane + Movable clear the gate under balanced CE → automatic #315 handoff → the
  margin law finally runs on an all-5-born field; (b) Lane still unborn → the CE-family Lane
  verdict hardens honestly toward FORMULATION for this init/objective FAMILY (unweighted AND
  balanced both measured) → Lane routes to the composed carriers already priced (d3a analytic
  Lane carrier · cb1 Lane band 1–2 KB) + a REVIEWED gate revision for a 4-class margin handoff.

## 5. Storage leg (the #1313 class, closed both sides)

- r4 consumed ~52 GB vs 23.3 GB projected: the projection under-modeled VERDICT payloads —
  `atomic_npz` used `np.savez` (zip STORED), 271 MB/verdict of uint8 camera + f16 logits.
  Measured deflate ratio 2.07×. Cures landed fc10f24637: (a) trainer writes
  `np.savez_compressed` going forward; (b) `tools/repack_npz_deflate.py` — lossless in-place
  repack with per-array sha256/dtype/shape verification + certify manifest rows
  (`npz_deflate_repack.v1`). r3 fully repacked (20/20 verified, manifest 20 rows); r4/r1/init
  repack chain running detached (counter 690) at memo time.
- Incident note: the r3 repack leg saw a benign concurrent-writer race (a reaper-orphaned
  foreground python + the detached chain shared a temp filename). Resolved harmlessly —
  independent 20/20 integrity re-check passed; lesson: scratch temp names should carry the PID.

## 6. Family state after r4

| leg | status | evidence |
|---|---|---|
| rate | PROVEN through training | ~107.5 KB repeat-identical re-encodes (r2/r3/r4) |
| pose | realization PROVEN ×3 | r4 pose_mse 0.002248 co-descended through the full CE window |
| seg  | 3 classes at gate; Lane+Movable = the residue | flip 0.01573 censored vs the 0.2504 frozen wall |

Gap to a claimable candidate remains large (flip 0.01573 vs the ~0.00116-class box; n32 single
seed; advisory axis). This memo claims a MECHANISM boundary (where unweighted CE saturates) and
a sealed cure, not a score.

## 7. Custody

- r4: `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/governed_n32_r4/`
  (200 ckpts, 200 verdict npz, cap checkpoint, launch manifest).
- r5 config: `.../AUTHORIZED_N32_R5_6000_20260828.json` sha 307b4e48e810796478d01e5a2fdaa55d6f950ef75ac81e783900071d45282ba1.
- §2 tables reproducible from `verdict_0200_step_001000.npz` with ~15 lines of numpy (executed
  this session).

## Observability surface
Per-layer: per-step history + per-5-step checkpoints + per-5-step verdict payloads (now
deflated). Decomposable: §2 per-class/per-axis; pose co-descent separately. Diffable: r4 vs r5
same schema, single-variable (weight mode). Queryable: JSON/npz on AP custody + repack
manifests. Citeable: config shas 4f5326a4c77b… (r4) / 307b4e48… (r5), launch counters 689/690.
Counterfactual: 5-step checkpoint cadence enables any-window replay; r5's balanced-vs-none A/B
is pre-built into the config gate.

— end —

---

## ADDENDUM (ddm_eq1, 2026-09-04) — the equations leg

**Law:** `logit_adjustment_class_prior_law_v1` — `tac.canonical_equations.logit_adjustment_class_prior_20260707` (`tac.canonical_equations`). **Relation:** DOMAIN-EXTENSION CANDIDATE — the law's PREMISE, measured.

Lane NEVER births under UNWEIGHTED CE (pred 0.0051% vs GT 0.60%, within-err 99.76%; Movable 27.84%). That is the exact failure the class-prior offset is Fisher-consistent against, measured here on the QBFLOW vehicle at a 1,000-step censored cap.

This memo's Catalog #344 trigger was the word **stratified** — `"ratified"` is a substring of it, and the gate matched plainly. MEASURED by this arm: 16 of the 29 live memos (55.2%) tripped the gate ONLY that way, i.e. the gate was flagging the memos that did their sampling right. Fixed in the same batch (`(?<!st)ratified`); the disposition above stands on its own merit, not on the misfire.
