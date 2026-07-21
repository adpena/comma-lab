---
schema: codex_premise_falsification.v1
task_id: "578"
lane_id: lane_realization_g2b_supportfill_578_20260721
axis: "[macOS-CPU advisory]"
score_claim: false
promotion_eligible: false
main_landing_review_required: true
---

# Premise falsification: G2b support-fill does not decode semantic cells to RGB

## Verdict

`SOURCE_RGB_CONTROL_EXACT_ZERO_BYTE_CELLS_TO_RGB_PREMISE_FALSIFIED`

The delegated premise said the canonical support-fill mapped target cells to a realizable RGB plane. Source inspection and real n600 measurement show a narrower contract: canonical support-fill accepts an already-specified uint8 RGB scorer plane and constructs its camera preimage. The real `seed_compose_b2` seed and 600 preserved stages carry HxW class IDs, not HxWx3 RGB planes. No receiver-side class-cell-to-RGB decoder or frame-0 synthesis rule exists.

This is a formulation-scoped handoff negative. It does not falsify exact factor-2 realization, support-fill, predict-project, or learned generator families.

## Measured evidence

The extended `tools/measure_realization_g2_lattice.py` ran the real n16/n64/n600 seed and a deliberately counted, encoder-supplied source-RGB control through the merged receiver, deterministic double decode, frozen native CPU-Torch SegNet, and PoseNet.

| rung | factor-2 byte-exact | semantic exact pairs | target d_seg | description d_seg | d_pose | added raw RGB bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1.0 | 0/16 | 0.0001325607 | 0.3478174210 | 0.0000408741 | 18,874,368 |
| 64 | 1.0 | 0/64 | 0.0001262824 | 0.3477973143 | 0.0000600439 | 75,497,472 |
| 600 | 1.0 | 0/600 | 0.0001518673 | 0.3434977214 | 0.0001016086 | 707,788,800 |

At n600, all 600 double decodes were identical and all 600 pose rows remained inside their declared tubes. Yet only 28 of 3,188 declared semantic writes survived in the source-RGB control, and zero pairs were semantically exact. The control therefore reproduces the known exact-plane anchor while disagreeing with the seed description by about 0.3435.

The 707,788,800-byte count is the fixed raw two-plane custody convention, not a compressed archive measurement. No contest score is claimed. The `0.1910828242 [contest-CPU]` pointer remains unchanged.

## D1-D4 disposition

- D1: exact RGB-plane projection/lattice is measured and deterministic; the seed path remains blocked before RGB.
- D2: n16/n64/n600 hard measurements landed; the requested zero-byte semantic realization fails.
- D3: n600 mean total time is 1.1320692461 seconds/pair; lattice double decode is 0.1755636077 seconds/pair; added raw source-plane bytes are 707,788,800.
- D4: `predict_project_realization_admissibility_v1` is registered with this false n600 anchor. It requires receiver-derived RGB, semantic exactness, deterministic exact lattice, pose-tube survival, and zero added seed bytes as one fail-closed conjunction.

## Triality

- DSL/code: no new vehicle lever was invented. The existing strict RGB custody boundary remains authoritative; label fields cannot impersonate RGB.
- DAG: `seed bytes -> class cells -> [MISSING receiver cells-to-RGB] -> RGB support-fill -> lattice -> camera` now blocks at the correct edge. The counted source control enters only after the missing edge.
- Equations: `predict_project_realization_admissibility_v1` preserves the failed conjunction as reusable system intelligence instead of promoting the exact downstream control.

## STORES CONSULTED

- Delegated authority prompt and the binding `CLAUDE.md` / `AGENTS.md` contracts.
- `reports/latest.md`, `.omx/state/lane_registry.json`, and `.omx/state/subagent_progress.jsonl` for pointer and ownership context.
- `.omx/research/realization_g2_lattice_receipt_20260721T085455Z.json` and its DAG/reuse artifacts.
- `.omx/research/seed_compose_b2_measurements_20260721.json` and the real SSD seed/stage custody.
- `src/tac/optimization/predict_project_receiver.py`, `uint8_lattice_feasibility.py`, `tie_aware_preimage.py`, and `resize_full_kernel.py`.
- `reports/tie_aware_preimage_ab_receipt_n600_fidelity.json`, #547/#549 exact-plane surfaces, and the source-derived n600 GT cache.
- `/Volumes/VertigoDataTier/pact/evidence/realization_g2b_20260721/receipt.json`, SHA-256 `ecea57bff31f468c7e3086b0391da5478e2c2379d0c4830af44d9eb638558363`.

## Reactivation criterion

Provide a deterministic, scorer-free receiver function that derives both RGB planes from counted seed bytes. Then rerun the same n16/n64/n600 admission ladder. Palette tables derived from the source, source planes smuggled into code, or encoder-only RGB custody do not close the blocker.

MAIN must review the code, evidence custody, false-anchor equation, and serializer commit before landing.
