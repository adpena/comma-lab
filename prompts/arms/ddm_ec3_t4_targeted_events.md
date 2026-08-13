# ddm_ec3 — gen-3 event alphabet: proposals aimed at the T4-CUSTODY flip map, priced by net-S

**Charter date:** 2026-08-13 · **Owner:** codex arm (xhigh) · **Consumer:** MAIN fires the proven vd1 T4 per-event validator (~$0.11/200 events, #381).
**Read first:** CLAUDE.md/AGENTS.md + `docs/operating_manual_craft_handoff.md` + `.omx/research/ddm_js1b_cuda_custody_adjudication_20260813.md` (incl. ADDENDUM) + `.omx/research/ddm_vd1_census_verdict_20260812.md` (incl. ADDENDUM) + `.omx/research/ddm_gv2_census_signal_decomposition_20260812.md`.

## Why gen-3 is not gen-1/gen-2 warmed over (the three banked cures composed)

Every prior event generation died on ONE of three now-measured defects; gen-3 is defined as the composition of all three cures:
1. **Phantom targeting (NEW, from js1b).** Gen-1/gen-2 proposed against the LOCAL Mac flip map, which carries
   +15,425 phantom flips on cp135 (+44%) and even INVERTS candidate signs. The T4-custody cp135-vs-GT
   disagreement map now exists on disk: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/ddm_js1b_20260813b/retained/fields/`
   (cp135_base_argmax_n600.npy + gt_argmax_n600.npy, 600×384×512 uint8, custody in the js1b receipt).
   Gen-3 proposals target ONLY pixels where the T4 field disagrees with T4 GT (34,970 real errors,
   Road-incident 81.6%; per-edge decomposition banked in `stage0_from_js1b/STAGE0_RESULT.json`).
2. **Per-axis selection (from cp5v).** The binding law: accept only events with predicted
   `100·Δd_seg + 603·Δd_pose < 0` (joint net S; the 603 S/unit pose marginal at base d_pose 6.88e-6).
   Never per-axis thresholds — the vd1 eligibility gate's measured defect.
3. **Advisory→exact precision 13% (census §3).** Divide any local optimistic projection by ~7.7× before
   comparing to the 0.000216 S fire bar, and design for the census's measured structure: pose costs on
   flip-positive events were 2–42× over budget NEAR-MISSES — minimize pose leak per event (smallest
   photometric amplitude that flips; prefer sites far from PoseNet-sensitive regions per the js4 Jacobian
   receipts), and carry ALL THREE edge families (Undrivable→Road · MyCar→Road · Road↔Lane), not one.

## The job

1. Build the gen-3 proposal producer against the T4-custody fields: candidate events = minimal EC1-wire
   token edits (the proven +0 B carrier, jo1/cp5v machinery) targeting T4-real flip clusters, ranked by
   predicted joint net S. Emit the vd1 store schema UNCHANGED (`event_store_target_anchored` layout) so the
   vd1 validator re-fires with zero adaptation. 200 events max, coverage across the full n600 heavy tail
   (g3 atlas), not the 6 pairs gen-1 reached.
2. Local prescreen ONLY for ordering, never for verdicts (the 13% law); the exact verdict is vd1's.
3. Pre-registered falsifier: if the top-200 gen-3 store's OPTIMISTIC additive eligible gain (net-S priced)
   is < 0.000216 S, the discrete-event family is FORMULATION-closed on cp135 for good — report and stop;
   MAIN will not spend the validator row.
4. No Modal/scorer/MPS from the arm. Land via serializer (post-edit working-tree shas) → final message =
   READY_TO_FIRE + the pinned vd1 dispatch command + projected K arithmetic, or the falsifier verdict.

## Custody
- Archives/fields READ-ONLY. P0 KEEP-THE-PAYLOAD on any generated store. Budget: build $0; the validator
  row (~$0.11–0.15) fires from MAIN only if the falsifier does not fire.
