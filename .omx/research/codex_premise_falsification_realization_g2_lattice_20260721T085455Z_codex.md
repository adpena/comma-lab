---
schema: codex_premise_falsification.v1
task_id: "578"
lane_id: lane_realization_g2_lattice_578_20260721
research_only: true
axis: macOS-CPU-advisory
verdict: COMPOSED_RGB_LATTICE_BUILT_SEED_TO_RGB_PROJECTION_PREMISE_FALSIFIED
verdict_scope: seed_compose_b2 projected class IDs to composed RGB factor-2 lattice handoff
score_claim: false
promotion_eligible: false
pointer_moved: false
main_landing_review_required: true
---

# Task 578: realization G2 lattice premise falsification

## Outcome

The strict composed RGB lattice stage is implemented in the existing receiver, but D1 did not land on the real n600 seed artifact. The preserved `seed_compose_b2` stages expose `predict_cell_field`, an HxW `uint8` class-ID field. They expose no HxWx3 `uint8` projected RGB plane, no hash-bound RGB generator custody, and no counted RGB payload. Passing that field to the existing factor-2 RGB solver would be a type error disguised as evidence.

The new receiver contract accepts only a genuine projected RGB plane plus hash-bound provenance. It reuses the existing exact factor-2 solver and full-kernel operator, verifies integer parse-back, and refuses either a 2-D label field or an encoder-supplied RGB plane falsely priced at zero bytes. A structural n2 fixture proves the composed RGB-to-camera operator can be exact with zero additional bytes when a decoder-derived RGB plane actually exists. That fixture is implementation evidence, not an n600 empirical claim.

## Measured prefix ladder

All rows are `[macOS-CPU advisory]`, score-ineligible, and measured from the preserved real stages. `null` means not attempted because the required RGB input was absent; it does not mean zero survival.

| n | label-only pairs | RGB pairs | lattice attempts | declared writes | class counts | stratum counts | exact fraction |
|---:|---:|---:|---:|---:|---|---|---|
| 16 | 16 | 0 | 0 | 97 | c0=42, c1=55 | boundary=97 | null |
| 64 | 64 | 0 | 0 | 337 | c0=151, c1=186 | boundary=337 | null |
| 600 | 600 | 0 | 0 | 3188 | c0=1379, c1=1807, c2=1, c4=1 | boundary=3183, critical=3, movable=2 | null |

The existing plane/cache replay remains separately true at n600: 600/600 cell descriptions are exact, 600/600 pose descriptions remain within their plane-level tubes, mean descriptive `d_seg=0.3434977213541667`, and mean tube debt `d_pose=0.0`. None of those values transfer to realized camera frames.

## Deliverable accounting

- D1: strict receiver-side RGB lattice callable landed; real n16/n64/n600 lattice attempts were blocked because the source RGB projection is absent.
- D2: decode time per pair and additional seed bytes are `null`; the zero-byte target is not met or disproved because no composed RGB frame exists. M2 remains an existence comparator only: exact through-camera realization at 1,717,172,741 counted bytes does not derive RGB from this seed.
- D3: realized-frame `d_pose` is `null`; plane-level `d_pose=0.0` is not promoted.
- D4: no canonical equation was registered because the required real n600 D1 anchor did not land.

The negative is scoped to the current label-to-RGB handoff formulation. It is not evidence against exact realization, which M2 already demonstrates with counted source RGB targets.

## Exact unblock

Supply a deterministic scorer-free decoder map that derives an HxWx3 `uint8` projected RGB plane from the seed, or supply and count the RGB payload. Bind the plane bytes, cell bytes, seed hash, generator identity, additional seed bytes, and zero decoder scorer invocations through `predict_project_rgb_plane_custody.v0`. Only then rerun n16 to n64 to n600 through the hard CPU-torch oracle and measure realized-frame pose.

## Stores consulted

- `CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`.
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`; `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`.
- `.omx/research/craft_handoff_witness_realization_ladder_20260720.md`.
- `.omx/research/m2_live_target_selection_20260720T1548Z.json`.
- `src/tac/optimization/predict_project_receiver.py`; `uint8_lattice_feasibility.py`; `joint_seg_pose_rate.py`; `resize_full_kernel.py`; `predict_project_schema.py`.
- `/Volumes/VertigoDataTier/pact/evidence/seed_compose_20260721/seeds/seed_compose_b2_loose.ppcs` and all 600 preserved hard-oracle stage JSON files.
- `.omx/state/lane_registry.json`; `.omx/state/subagent_progress.jsonl`; `.omx/state/canonical_task_status.jsonl`; `reports/latest.md`.
- Operator broadcasts through `2026-07-19T19:48:01Z`, including Fisher/margin ranking, corrected inner-Jacobian realization prediction, curvelet/shearlet basis, and pose factorization via xi. No Fourier basis was introduced.

## Custody and pointer delta

The SSD receipt is `/Volumes/VertigoDataTier/pact/evidence/realization_g2_20260721/receipt.json`, SHA-256 `7e3723280833963a7fb9a1449ecacfaaf1746bcb497320af8e4687f3118c4496`. The durable repository receipt is `realization_g2_lattice_receipt_20260721T085455Z.json`. The frontier pointer remains `0.1910828242 [contest-CPU]`; no score, GO, launch, or promotion claim is made. MAIN must review this branch diff before landing.
