# ddm_qbt2b r3 CE-birth verdict — the seg wall is BROKEN: realized flip 0.5165 → 0.0680 in 100 CE steps (3.7× below the r2 0.2504 wall), 4 of 5 classes BORN incl. Road; Lane alone unborn; cap_without_handoff by design; r4 extended-window continuation FIRED

STORES CONSULTED: qbt2b charter (`.omx/research/charters/ddm_qbt2b_inherited_palette_birth_20260828.md`)
· qbt2b deliverable (2df4cce961) · sealed r3 fire order (8ae8fcf2) · qbt1 r1+r2 verdict + erratum
(269566a151) · #315/#686 derived-schedule law (CE births, margin sharpens) · m194 paint-not-partition
· m131 Lane-demand law · #208 rare-class init · m110 pose budget · m143 cross-regime transfer ·
charter-provenance memory (file::symbol pins).

score_claim=false everywhere. All rows [macOS-Metal/CPU frozen-scorer advisory, n32
seeded-stratified]. Pointer UNMOVED (gb1 0.14811799921260607).

## 1. The run (MEASURED)

- Fired 2026-08-27 (launch counter 688, pid 23594) per the sealed fire order after MAIN's
  full chain: 2-pass .py review · 4/4 sealed-hash verification · live AP df 61 GiB >
  36.94 GB demand · Metal slot free · lanes claimed · authorized config validate-config
  PASS (sha fa9abaf66d65…).
- Ran 471 s, rc=0, clean exit at the 100-step birth SAFETY cap: `stage_03a_cap_without_handoff.pt`
  saved, margin stage correctly REFUSED (the #315 event gate requires ALL 5 classes at
  within-class error <0.20 for 2 consecutive realized verdicts). 20 verdicts + 20 periodic
  checkpoints + per-checkpoint re-encodes all landed; verdict payloads RETAINED
  (camera/logits/argmax/pose/targets per pair — the P0 payload law).
- MEMORY-GATE INCIDENT (m143, recorded): the first fire was REFUSED by the SUM-over-RAM
  admission gate at the sealed 86.0 GiB declaration — whose binding max-leg was the WD3
  85.76-GiB CROSS-VEHICLE scorer precedent. Cure: re-declared at the r2 same-vehicle
  measured anchor (52 GiB; r2 child RSS ~1 GB over 11,138 s at identical chunk-16/n32
  geometry). Gate never weakened; safe_run 116 GiB hard cap retained. r3 child peak RSS
  observed ~6.3 GB — the 52 GiB declaration is ~8× conservative on this vehicle.

## 2. THE HEADLINE (MEASURED, from retained verdict payloads)

The r2 seg wall — realized flip FROZEN at 0.2504 for 4,670 consecutive steps under the
expected-flip-margin law — is BROKEN by the CE birth stage:

| step | realized flip | pose_mse | classes present (pred > 0.01%) |
|---|---|---|---|
| 5   | 0.5165 | 128.3 | Road·Undriv·MyCar |
| 50  | 0.1757 | 61.2  | +Movable |
| 100 | 0.0680 | 0.681 | Road·Undriv·Movable·MyCar |

Per-class at step 100 (32 pairs, 512×384 realized argmax vs target):

| class | GT area % | predicted % | within-class err % |
|---|---|---|---|
| Road       | 23.12 | 21.40 | 16.78 (falling) |
| Lane       |  0.60 |  0.00 | 100.00 (never predicted) |
| Undrivable | 49.58 | 50.17 | 2.74 |
| Movable    |  1.25 |  0.43 | 77.43 (falling) |
| MyCar      | 25.46 | 28.00 | 0.00 |

- Flip was still descending steeply at the cap (0.0741 → 0.0680 over the last 5 steps) —
  0.0680 is a CENSORED bound, not an asymptote.
- Pose co-descended THROUGH birth (128.3 → 0.681), tracking the r2-proven realization
  trajectory shape — the [INFERRED] pose–seg interior conflict from the r1/r2 memo did
  NOT materialize as a birth blocker at this window.

## 3. Adjudication vs the charter's pre-registered falsifier

- FALSIFIER ("inherited-palette init + readout fit + CE still cannot birth Road at n32"):
  **NOT FIRED.** Road is born and near its GT share. The PRIOR-LAW PREDICTION half is
  CONFIRMED at this window: the #315/#686 law (CE births, margin sharpens) is now
  MEASURED on this vehicle — the same field the margin law held frozen for 4,670 steps
  moved 7.6× in 100 CE steps.
- verdict_scope: INSTANCE (qbt2b stage-03a at n32, single seed #1251, 100-step window)
  for every number above; the LAW confirmation (CE-births-where-margin-cannot) is a
  second measured anchor for #315/#686, not a new law.
- The residue is exactly ONE class: Lane (0.60% area — the campaign-wide hardest orbit,
  m131: Lane = 90.1% of the rate-representation demand on the frontier body). Lane's
  100.00% within-class error across all 20 verdicts is the entire distance between
  cap_without_handoff and the margin-stage handoff.

## 4. The continuation (FIRED)

- r4 (launch counter 689, pid 37053, lanes ddm_qbt2b_r4_metal/scorer_20260827):
  extended birth window 100 → 1,000 + the unchanged margin+pose stage (5,000) on event
  handoff. Fresh compiler-derived config (validate-config PASS sha 4f5326a4c77b…; EMA
  LawRef re-resolved 0.9992327661102197 @ 6,000 updates). NOT a resume: the checkpoint
  `config_identity` deliberately binds `birth_max_steps` and REFUSED the extension —
  the identity guard held and was not weakened; re-running the ~8-minute prefix is the
  price of integrity. Storage 23.3 GB ckpts + tars vs 60.1 GB AP free (PASS).
- r4 forks: (a) Lane births within 1,000 CE steps → automatic #315 handoff → the margin
  law runs 5,000 steps on an all-5-born field — the first chance this family has ever
  had at a real seg asymptote; (b) Lane still unborn at 1,000 → the CE-alone Lane
  verdict hardens to FORMULATION scope for THIS init/objective, and Lane routes to the
  composed-carrier alternatives already priced in the corpus (d3a analytic Lane carrier;
  cb1 Lane band 1–2 KB) while the 4-class field proceeds to margin via a gate revision
  (a DELIBERATE code change with review, not a silent relaxation).
- Stage 05 remains BLOCKED pending the real same-budget QBW1 control (unchanged).

## 5. Family state after r3

| leg | status | evidence |
|---|---|---|
| rate | PROVEN through training | ~107.5 KB repeat-identical across 973+20 re-encodes |
| pose | realization PROVEN ×2 | r2 119.84→4.5e-4; r3 co-descent 128.3→0.681 in 100 steps |
| seg  | WALL BROKEN, Lane residue | flip 0.0680 censored-descending vs the 0.2504 frozen wall |

No family in this campaign has previously held all three legs live on one vehicle. The
gap to a claimable candidate remains large (flip 0.0680 vs the ~0.00116-class box; n32
single-seed; advisory axis) — this memo claims a MECHANISM, not a score.

## 6. Custody

- r3: `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/governed_n32_r3/`
  (20 ckpts, 20 verdict npz w/ full payloads, re-encodes, cap checkpoint, launch manifest
  incl. the REFUSED-then-cured memory declaration record).
- r4: `.../governed_n32_r4/` + `AUTHORIZED_N32_R4_6000_20260827.json` (sha 4f5326a4c77b…).
- §2 tables reproducible from the retained `verdict_00NN_step_*.npz` with ~15 lines of numpy.

## Observability surface
Per-layer: per-step history rows + per-5-step checkpoints + per-5-step verdict payloads.
Decomposable: §2 per-class/per-axis. Diffable: r2 vs r3 vs r4 same schema; r4's first 100
steps vs r3 (EMA-law delta only). Queryable: JSON/npz on AP custody. Citeable: config shas
fa9abaf66d65… (r3) / 4f5326a4c77b… (r4), launch counters 688/689. Counterfactual: 5-step
checkpoint cadence enables any-window replay.

— end —

---

## ADDENDUM (ddm_eq1, 2026-09-04) — the equations leg

**Law:** `logit_adjustment_class_prior_law_v1` — `tac.canonical_equations.logit_adjustment_class_prior_20260707` (`tac.canonical_equations`). **Relation:** DOMAIN-EXTENSION CANDIDATE (law's lever is the level-set witness trainer; QBFLOW is a different vehicle).

Realized flip 0.5165 → 0.0680 in 100 CE steps births 4 of 5 classes and leaves Lane unborn — the rare-class failure this law's zero-byte τ·log(prior_c) offset exists to cure. r4/r5 carry the premise and the cure measured on this same vehicle.

This memo's Catalog #344 trigger was the word **stratified** — `"ratified"` is a substring of it, and the gate matched plainly. MEASURED by this arm: 16 of the 29 live memos (55.2%) tripped the gate ONLY that way, i.e. the gate was flagging the memos that did their sampling right. Fixed in the same batch (`(?<!st)ratified`); the disposition above stands on its own merit, not on the misfire.
