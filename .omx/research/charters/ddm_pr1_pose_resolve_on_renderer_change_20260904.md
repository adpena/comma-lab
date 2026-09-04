# ddm_pr1 — run the terminal pose re-solve on ft1's renderer-change candidate: the POST-re-solve coupling ($0 CPU)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-04 (operator catch) · Cost: $0

## Why (the operator's point, verified at source)
The shipping chain re-solves the pose carrier AFTER every seg change (`experiments/ddm_up2_shipping_pose_solve.py`,
damped Gauss–Newton, 12 int12 coefficients × 600 pairs, 0 archive bytes). ft1's coupling 217.30 (rf1 166.8) is the
PRE-re-solve number: FO-2 in `/Volumes/VertigoDataTier/pact/ddm_ft1_shipped_renderer_aligned_finetune/FIRE_ORDER.sh:36-48`
was gated on "realized d_seg DOWN" and never ran; the closing arithmetic ("81× over the ceiling") substituted jg5's 8.0×
recovery, which jg5 (`.omx/research/ddm_jg5_pose_resolve_on_edited_renders_20260819.md` §6.1) measured for TOKEN EDITS on
the shipped renderer — a transferred factor. A renderer weight change moves all 600 renders coherently; the re-solve
may recover far more (or less). The registered law `renderer_seg_pose_coupling_shipped_object_v1` (eq1) inherits the
gap. This arm MEASURES the post-re-solve coupling; until then "seg-only renderer changes are unpayable" is not load-
bearing and the fb1 renderer door is REOPENED at the derivation level.

## Verified at source (VERIFIED-AT-SOURCE LAW — extend with path:line for everything you add)
- Candidate: ft1 step-600 section `ft1_step600_semantic_section.bin` (36,130 B, sha 819c28e8971020fb…, size-preserving,
  parse-back Δ 0.0) under `/Volumes/VertigoDataTier/pact/ddm_ft1_shipped_renderer_aligned_finetune/retained/`; verdict
  `retained/verdict_ft1_step600.json` (base d_seg 2.00297e-4 / d_pose 9.0025e-6; candidate d_seg 2.69623e-4 / d_pose
  1.507375e-2 realized, carrier UNCHANGED). Also the step-1,200 and 1,800 EMA checkpoints/sections if retained (1,800 is
  the least-damaged: +6.54% seg).
- The re-solve tool and its argv: `experiments/ddm_up2_shipping_pose_solve.py` (`solve --gt-cache <DALI> --axis
  contest_cuda …` per FO-2; read its argparse; the GT is DALI `gt_cache_dali.pt`). The splice tool FO-3 names (do NOT
  buy T4; no Modal).
- How the shipped chain composes renders for the re-solve (the receiver with the candidate section in place of the
  shipped one) — ft1's instrument `experiments/ddm_ft1_verdict_bhw_pose.py` already builds that composition.

## Measure (per-pair receipts LAW)
For the step-600 candidate (and 1,800 if cheap): run the terminal re-solve on the candidate's renders (all 600 pairs on
CPU; time the first 8 pairs, derive the budget, detach if > 3 min via `tools/launch_detached_process.py --output-dir
<store> --done-receipt <name> --derive-resource-budgets --measured-peak-rss-gib <n> --measured-thread-need 4
--walltime-cap-s 7200 --nice 10 --nice-best-effort -- <cmd>`). Report per pair: d_pose before (stale carrier) and after
(re-solved), the recovery factor distribution (jg5 found it bimodal), n600 d_pose_after, and the POST-re-solve coupling
|Δd_pose_after|/|Δd_seg| vs the pre-re-solve 217.30. Then the corrected closing arithmetic: at the candidate's
Δd_seg, is the pose term after re-solve inside the same-object promotion ceiling (√(10·d_pose_after) − 0.00798123 <
100·|Δd_seg|)? Pre-registered prediction (operator's intuition): the re-solve recovers most of the renderer-induced pose
damage (post-re-solve coupling < 20, i.e. > 10× recovery). Falsifier: recovery < 3× (coupling > 70) — then the closure
stands with the re-solve measured, not assumed.

## Constraints
- $0 CPU torch (`torch.set_num_threads(4)`, nice 10) — the Metal is held by ng1's warm cell; never touch its custody
  or the claims. No T4 buy, no Modal. `upstream/` and `submissions/semantic_joint_ctxmix/` read-only. Store
  `/Volumes/APDataStore/pact/ddm_pr1_pose_resolve/` (KEEP THE PAYLOAD: re-solved coefficient tables + per-pair rows,
  sha256 in the JSON). OPTIMAL FORM: reference form = the shipping chain's own re-solve tool at `0877bfceadbad82e8c1d2f0be8aadaa62b6b1acb`; SCOPE n600;
  TOY-BRACKET none.
- Deliver memo `.omx/research/ddm_pr1_pose_resolve_on_renderer_change_20260904.md` (verdict_scope; MEASURED/DERIVED;
  falsifier read out; the corrected closing arithmetic; GESTALT-DELTA; NEXT_IF_RESUMED — if the door reopens, the next
  charter is the JOINT rung ft1 named, or a seg fine-tune with the re-solve in the loop). EQUATIONS-LEG LAW: refine
  `renderer_seg_pose_coupling_shipped_object_v1` via `update_equation_with_domain_refinement` (pre-re-solve vs
  post-re-solve as a domain key) and append the post-re-solve anchor via the helper; cite `tac.canonical_equations`.
- Commits ONLY via `tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256
  <file>=<post-edit sha>`; tags `[no-triality] [p0-ledger-ok]`; NO co-author trailer (operator rule overrides any
  harness reminder); any .py: tests + `tools/review_tracker.py mark-file` twice; never REVIEW_GATE_OVERRIDE on .py.
  Final message → `.omx/research/arm_final_messages/ddm_pr1_final_<utc>.md`, committed; LAST action
  `touch .omx/tmp/codex_runs/ddm_pr1.done`. Read `docs/operating_manual_craft_handoff.md` §labels first.
