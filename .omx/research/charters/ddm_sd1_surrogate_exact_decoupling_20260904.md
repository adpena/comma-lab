# ddm_sd1 — where does the expected-flip surrogate mis-price the exact argmax? ($0 read on retained QBR1 milestones)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-04 · Cost: $0 (CPU; the Metal is
held by the live chain)

## Why
QBR1 seed 20260902, both cells: `seg_expected_flip_realized` (the training surrogate) fell monotonically 0.005018 →
0.003254 (−35.1%) while the exact `d_seg_hat` rose to a peak at step 2,000 and ended +9.56% above its start (ng1
d57e49b02, memo `ddm_ng1_warm_transition_burn_design_20260904.md`). ft1 showed the same decoupling (loss −72% while
d_seg +31% → +6.5%). If the surrogate mis-prices the exact argmax, the born trainer is optimizing the wrong object and
vr1 rows 1 (hard-site margin weight), 3 (area cap), 4 (per-edge τ from the rank-4 head) are calibration cures — but
WHICH one depends on WHERE the mis-pricing lives (class, edge, margin band, pair). This arm measures that, with
receipts, so the next generation's second race is chosen on a decomposition rather than a guess.

## Verified at source (VERIFIED-AT-SOURCE LAW — extend with path:line for every premise you add)
- Retained milestones (read-only custody of the live chain): `/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/runs/
  seed_20260902/{control_native100,treatment_zero_native}/milestones/step_00{0..5}000/` — find at source what each
  milestone retains (EMA state? live state? rendered argmax? camera uint8?) and whether the exact per-pixel argmax at
  the milestone can be REPRODUCED bit-for-bit from it (ar1 did this on `stage_01_end.pt` through the sealed trainer's
  own forward + `roundtrip_to_camera_uint8_ste` + frozen CPU-torch SegNet: `experiments/ddm_ar1_aa_render_price.py`
  is the working instrument — reuse it, do not rebuild).
- Surrogate definition: `experiments/ddm_qbt1_qbflow_trainer.py` `expected_flip_margin_loss` (:523-538 region; scalar
  τ `tau_for_step` linear 0.15→0.05 at :622-626) and how `seg_expected_flip_realized` is computed per step (history.jsonl).
- GT authority DALI `gt_cache_dali.pt`; the burn's own frame is PyAV `gt_n600.npz` — report both, DALI is authority.
- The trained selection is `qbt.SELECTION_IDS` (n32); the other 568 pairs are UNFITTED — read both populations, never
  mix them (ar1's dose-response: trained 1.314× vs unfitted 1.118×).

## Measure (per-pair receipts LAW)
For milestones 0, 1k, 2k, 5k of the control cell (treatment if cheap): per pair and per (GT class, runner-up class)
edge: exact flip count (argmax vs DALI lstars), the surrogate's expected-flip mass at that milestone's τ, and their
ratio; the same split by GT-margin band (annulus |margin| < δ_R 0.021882 vs interior). Deliver: (a) the mis-pricing
map — which edges/bands carry surrogate mass that never flips, and which flips carry no surrogate mass; (b) the sign of
Δ(exact) vs Δ(surrogate) per edge between 0→2k and 2k→5k; (c) a one-line recommendation ranking vr1 rows 1/3/4 by
the fraction of the exact excursion each would have priced correctly, DERIVED from the map. Pre-registered prediction
(from vr1 row 4): the Undrivable↔Movable and Road↔Lane edges are mis-scaled ≥2× by the scalar τ; falsifier: the
per-edge ratio spread across edges < 1.3× (then τ is not the defect and row 1's spatial weight is the next race).

## Constraints
- $0, CPU torch, `torch.set_num_threads(4)`, nice 10; anything > 3 min via `tools/launch_detached_process.py --output-dir
  <store> --done-receipt <name> --derive-resource-budgets --measured-peak-rss-gib <n> --measured-thread-need 4
  --walltime-cap-s 3600 --nice 10 --nice-best-effort -- <cmd>`. Never write under the chain's `runs/`; never touch the
  Metal or the claims. Store `/Volumes/APDataStore/pact/ddm_sd1_surrogate_decoupling/` (KEEP THE PAYLOAD: per-pair
  rows + per-edge tables + at least 4 rendered argmax arrays per milestone, sha256 in the JSON).
- OPTIMAL FORM: reference form = the sealed trainer's own forward/roundtrip/scorer path (ar1's instrument) at commit
  `07fa60f3697f39d7e9c04232ea9d689f02406b32`; SCOPE = 4 milestones × 32 trained pairs (+ unfitted if cheap); no mechanism reduction; TOY-BRACKET none.
- Memo `.omx/research/ddm_sd1_surrogate_exact_decoupling_20260904.md` with verdict_scope, MEASURED/DERIVED labels,
  the falsifier read out, GESTALT-DELTA line, NEXT_IF_RESUMED; EQUATIONS-LEG LAW: cite `tac.canonical_equations`
  `scalar_top1_top2_margin_is_exact_distance_to_flip_v1` / `segnet_head_rank4_linear_flipdist_v1` (append an anchor via
  the helper if the map fits their domain; else FORMALIZATION_PENDING naming the law it would need).
- Commits ONLY via `tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256
  <file>=<post-edit sha>`; tags `[no-triality] [p0-ledger-ok]`; NO co-author trailer (operator rule overrides any
  harness reminder); any .py: tests + `tools/review_tracker.py mark-file` twice; never REVIEW_GATE_OVERRIDE on .py.
  Final message → `.omx/research/arm_final_messages/ddm_sd1_final_<utc>.md`, committed; LAST action
  `touch .omx/tmp/codex_runs/ddm_sd1.done`. Read `docs/operating_manual_craft_handoff.md` §labels first.
