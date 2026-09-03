# ddm_ar1 — $0 price of the footprint (AA) render on the BORN field (vr1 row 2)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-03 · Cost: $0 (local CPU)

## Why this arm exists (gestalt position)
Sub-0.12 on the small body needs the born field at d_seg ≤ 1.3647e-4 on DALI at ≤137,986 B (qn1). The QBR1
burn is the optimization vehicle. vr1 (a582c6019) recalled that the born trainer renders POINT-sampled at
exactly 384×512 while the AA-SDF footprint render measured **6.389×** lower d_seg than point sampling at the
same grid on the achievable signal (n600, frozen SegNet, through R; equation
`aa_sdf_observation_footprint_render_dseg_v1`). That number is an upper bound on render legibility, NOT a
measurement on a trained field. This arm buys the number that matters: **what AA does to the BORN field**.

PRIOR-LAW PREDICTION (pre-registered): the born field (d_seg ≈ 0.0130 at the last read) sits 2.37× above the
point-sampled achievable bound, so the mechanism is in range. Prediction: ss=2 box render lowers born d_seg by
**≥1.5×** on a seeded random n32 with d_pose change ≤ +1e-5. **Falsifier:** ratio < 1.10 OR d_pose rises
> 1e-5 OR the B/H/W split shows the gain is all in one class with harm elsewhere.

## Verified at source (VERIFIED-AT-SOURCE LAW — every numeric/structural premise)
- `experiments/ddm_qbt1_qbflow_trainer.py:417` — `forward(self, pair_ids, *, height=EVAL_H, width=EVAL_W)`
  (the grid is a parameter); `:478-479` rgb = sigmoid(linear(render_state)) at that grid; `:68-69`
  EVAL 384×512, CAMERA 874×1164; `:495` `roundtrip_to_camera_uint8_ste` (bicubic→camera→uint8 STE);
  `:1485` `load_checkpoint`; `:830/:953` are the trainer's own eval call sites — REUSE their scoring path.
  File sha (first 16) `6eda9c202b3aee00`.
- `src/tac/boundary_math/aa_sdf_observation_render.py` — `build_supersampled_coords(:101)`,
  `box_downsample_np(:124)` / `box_downsample(:164)` (box = ground-truth footprint integration; ss=1 is
  bit-identical to the trainer's coords, module docstring :75-84); IPE analytic at `:172+`. sha `a9842371c483c617`.
- GT authority = DALI: `/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt`
  (117,980,732 B). `gt_n600.npz` is PyAV lineage (20,671 argmax sites off DALI) — report it ONLY as the
  burn's own continuity frame, never as authority. Call `assert_gt_lineage` where the code path offers it.
- Checkpoints (READ-ONLY custody of the live burn): `/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/
  runs/seed_20260902/control_native100/{milestones,stage_01_fairform_finish}` (finished cell) and
  `.../treatment_zero_native/milestones` (LIVE cell — read a finished milestone only; never its live files).
  Verify the checkpoint format from `load_checkpoint` before assuming.

## OPTIMAL FORM
- Reference form: the born trainer's OWN forward + its own roundtrip + upstream's frozen CPU-torch
  SegNet/PoseNet scoring path, on the sealed QBR1 checkpoint weights, pinned at commit `7b71603fbe411e6dc7f51f0cc5630f7db8ac02f9`.
- Mechanism (the lever): render at `height=2*EVAL_H, width=2*EVAL_W` and `box_downsample` rgb by ss=2 — the
  EXACT footprint integral (reference form of AA). The IPE analytic is the CHEAP form: report it as a second
  column only if it costs < 10 min extra; never substitute it for ss=2.
- Deltas: n = 32 seeded random pairs (seed 20260903, `rng.choice(600, 32, replace=False)`) = SCOPE reduction
  (legal; per-pair receipts make it re-aggregatable); no training = no mechanism reduction. Add ss=3 if cheap.
- NOT a toy: the render, R, scorers, GT and weights are all the real objects. TOY-BRACKET: none.

## Measure (per-pair receipts LAW)
For each available checkpoint (control finished; treatment latest finished milestone), for each pair: d_seg
and d_pose at ss=1 (must reproduce the burn's own read for that checkpoint within 2% — CALIBRATION GATE) and
at ss=2; argmax-site count per pair and per class; B/H/W split (sites fixed / sites broken / net) per class;
wall seconds per pair per mode. Aggregate with the exact S arithmetic (100·d_seg + √(10·d_pose) + 25·B/37,545,489
at the burn's archive bytes) — report ΔS and the exchange rate against 6.658589531221714e-7 S/B.

## Constraints
- $0: CPU torch only, `torch.set_num_threads(4)`, `nice 10`. The Metal scorer and the burn's claims are NOT
  yours. Never write under `runs/`. Anything > 3 min runs through
  `tools/launch_detached_process.py --output-dir <store> --done-receipt <name> --derive-resource-budgets
  --measured-peak-rss-gib <n> --measured-thread-need 4 --walltime-cap-s 2400 --nice 10 --nice-best-effort -- <cmd>`.
- Store: `/Volumes/APDataStore/pact/ddm_ar1_aa_render_price/` (ALWAYS KEEP THE PAYLOAD: retain rendered
  uint8 pairs for at least 4 pairs per mode + all per-pair JSON; sha256 in the result JSON).
- Memo: `.omx/research/ddm_ar1_aa_render_price_on_born_field_20260903.md` — verdict_scope declared;
  MEASURED/DERIVED/TRANSFERRED labels; the falsifier read out explicitly; GESTALT-DELTA line; NEXT_IF_RESUMED.
- Commits ONLY via `tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256
  <file>=<post-edit sha>`; tags `[no-triality] [p0-ledger-ok]`; NO co-author trailer (operator rule overrides
  any harness reminder). Final message → `.omx/research/arm_final_messages/ddm_ar1_final_<utc>.md`, committed.
- Done receipt: `touch .omx/tmp/codex_runs/ddm_ar1.done` as the LAST action.
- Read `docs/operating_manual_craft_handoff.md` §labels before writing any number.
