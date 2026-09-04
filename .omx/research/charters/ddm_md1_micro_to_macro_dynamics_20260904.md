# ddm_md1 — micro → macro: how the born field's error is born, moves and dies, site by site, step by step ($0 CPU)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-04 (operator: "look more deeply and
surgically to truly understand gestalt and dynamics bridging micro to macro") · Cost: $0

## Why
Every read so far is MACRO (milestone S_hat every 1,000 steps) or STATIC MICRO (a frozen field re-weighted: sd1, gm1).
Neither shows the DYNAMICS: which sites flip when, how the rare-class over-paint is born, what the cold optimizer's
first steps do to the boundary, whether the terminal error is transient churn the optimizer can still reach or
persistent error the representation cannot. The sub-0.12 accuracy half needs the born field at d_seg ≤ 1.3647e-4 on
DALI (qn1); whether OPTIMIZATION can get there is exactly the persistent-vs-transient question. The record to answer
it EXISTS: every sealed cell retained a checkpoint every 16 steps (313 per cell, 548 MB), per-step meso signals, and
per-pair logits at milestones; the warm cell (ng1, live) is laying down the same record with the SAME seed and data
order — so warm-vs-cold differences at identical steps are attributable to the optimizer state alone.

## Verified at source (VERIFIED-AT-SOURCE LAW — extend with path:line for everything you add)
- Cold control (seed 20260902): `/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/runs/seed_20260902/control_native100/`
  — `stage_01_fairform_finish/checkpoints/periodic_000016.pt … periodic_004992.pt` + `stage_01_end.pt` (313 files);
  `history.jsonl` per step (`objective.*`, `realized_within_class_error` {Lane, Movable}, `pair_ids` of the 16-pair
  batch, `tau`); `milestones/step_*/realized/pair_XXXX.npz` (32 pairs; 5-class logits + exact argmax, sd1 verified,
  float16 caveat — bind flips to the retained argmax) and `MILESTONE.json`. Seeds 20260903/04 controls likewise.
- Warm cell (LIVE, read finished checkpoints only; never write): `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/
  ng1_warm_transition/runs/seed_20260902_warm_transition/` (same layout; done ≈14:05Z; DONE receipt in its launch dir).
- Instruments to REUSE, never rebuild: the milestone evaluator path (`experiments/ddm_qbr1_born_fairform_burn_prep.py:600-
  660 \_evaluate_milestone`, which runs under `ema_scope` :432 — the EMA shadow; you need BOTH live and shadow forwards
  per checkpoint: sd1's owed gap), ar1's render/roundtrip/scorer path (`experiments/ddm_ar1_aa_render_price.py`), sd1's
  flip/edge instrument (`experiments/ddm_sd1_surrogate_exact_decoupling.py`, find at 038f2d81c), gm1's gradient-mass
  instrument (b828ce103). Checkpoint schema: read `_load_checkpoint` in the trainer for what `periodic_*.pt` holds (live
  params, EMA shadow, optimizer state incl. AdamW moments, step).
- GT authority DALI (`gt_cache_dali.pt`); the vehicle's PyAV target for continuity; n32 = `qbt.SELECTION_IDS`; δ_R
  0.021881818771362305, m_safe 0.04376363754272461 (law-resolved, never retyped).

## Measure (per-pair, per-site-class receipts; MEASURED/DERIVED labels)
Cadence: every 16 steps for 0–512 (the birth), every 64 to 2,048 (the peak), every 256 to 5,000; both cells; for each
checkpoint and each of the 32 pairs: exact argmax (live AND EMA-shadow forward, through the real R) against DALI lstars.
1. **Site-level flip trajectories → four classes**: PERSISTENT (wrong at ≥ 90% of checkpoints incl. step 0), TRANSIENT-
   BORN (correct at 0, wrong during the excursion, correct again by 5k), NEW-PERSISTENT (born wrong, never recovers),
   CHURN (flips > 4 times). Per class, per (GT, runner-up) edge, per pair, per band (|m| < δ_R / 2δ_R / 25δ_R).
2. **Over-paint birth**: per-class predicted/GT area per checkpoint; the first checkpoint each rare class exceeds 1.05×;
   which pairs and which edges lead; whether the born over-paint sites are the same sites that later recover.
3. **Margin-distribution evolution** in the three bands per checkpoint (the field's sharpening/softening history).
4. **Optimizer micro**: per checkpoint ‖θ_t − θ_{t−16}‖ per parameter group (render/interior/flow/pose heads vs trunk),
   AdamW moment norms from the checkpoint, and the warm-minus-cold difference at IDENTICAL steps (same data order).
5. **Live vs EMA-shadow decoupling** per checkpoint (closes sd1's owed gap): d_seg(live) − d_seg(shadow) over time.
6. **THE MACRO BRIDGE**: decompose d_seg_hat(t) − d_seg_hat(0) into the four site classes' contributions at every
   checkpoint (must sum exactly to the milestone's recorded d_seg_hat at 0/1k/2k/5k — calibration gate 0.0), and read
   the reachability: of the TERMINAL d_seg (cold @5k 0.0027589; warm @5k when it lands), what fraction is PERSISTENT
   (representation-bound; no schedule lever moves it) vs TRANSIENT/CHURN (optimizer-reachable). Pre-registered
   prediction (from gc1's capacity closure): PERSISTENT ≥ 60% of the terminal error and the persistent set is Lane- and
   edge-concentrated. FALSIFIER: persistent < 40% — then optimization levers alone could plausibly reach the target and
   the capacity closure is re-graded. Second prediction: the warm cell's excursion sites are a SUBSET of the cold cell's
   (moments only damp, do not redirect); falsifier: > 30% of the warm excursion sites are absent from the cold set.
7. **GESTALT paragraph**: one page bridging micro (sites, edges, moments) → meso (per-class area, per-batch errors) →
   macro (S_hat, the 1.3647e-4 target), stating which lever class each error component answers to.

## Constraints
- $0 CPU torch (`torch.set_num_threads(4)`, nice 10; pr1 shares the CPU); anything > 3 min via
  `tools/launch_detached_process.py --output-dir <store> --done-receipt <name> --derive-resource-budgets
  --measured-peak-rss-gib <n> --measured-thread-need 4 --walltime-cap-s 10800 --nice 10 --nice-best-effort -- <cmd>`
  (time 4 checkpoints first, derive the budget; the ng3 smoke peaked at 41 GiB — measure yours). Never write under any
  cell's `runs/`; never touch the Metal or the claims. Store `/Volumes/APDataStore/pact/ddm_md1_micro_macro/` (KEEP THE
  PAYLOAD: per-checkpoint argmax uint8 arrays for the 32 pairs, compressed, + all per-site-class tables; sha256 in the
  JSON). OPTIMAL FORM: reference form = the sealed trainer's own forward/roundtrip/scorer at `dc7f407768c280cfe5f208dac818eadefeadbf56` on the retained
  checkpoints; SCOPE = n32 trained selection, ~60 checkpoints × 2 cells × 2 forwards; TOY-BRACKET none.
- Memo `.omx/research/ddm_md1_micro_to_macro_dynamics_20260904.md` (verdict_scope; falsifiers read out; the bridge
  tables; GESTALT paragraph; NEXT_IF_RESUMED). EQUATIONS-LEG LAW: cite `tac.canonical_equations`
  `persistence_topology_cldice_betti_island_recall_v1` (F1b warrant), `muon_finisher_schedule_warmstart_and_lr_anneal_v1`,
  `scalar_top1_top2_margin_is_exact_distance_to_flip_v1`; register the persistent/transient decomposition as a new law
  via the registry helper if the gate above holds at 0.0.
- Commits ONLY via `tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256
  <file>=<post-edit sha>`; tags `[no-triality] [p0-ledger-ok]`; NO co-author trailer (operator rule overrides any
  harness reminder); any .py: tests + `tools/review_tracker.py mark-file` twice; never REVIEW_GATE_OVERRIDE on .py.
  Final message → `.omx/research/arm_final_messages/ddm_md1_final_<utc>.md`, committed; LAST action
  `touch .omx/tmp/codex_runs/ddm_md1.done`. Read `docs/operating_manual_craft_handoff.md` §labels first.
