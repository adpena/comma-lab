# ddm_fs1 — the first exact-row candidate of the wave: per-pair frame-0 selector re-selection on afr1, byte-closed to a T4 row

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-04 · Cost: $0 local + ONE governed T4
buy (`tools/fire_modal_auth_eval.py`, ≈$0.15–0.30) fired by MAIN after the arm seals

## Why (pr1, MEASURED on the LIVE afr1 object)
pr1 (c7b537053; memo `.omx/research/ddm_pr1_pose_resolve_on_renderer_change_20260904.md`) swept the receiver's per-pair
frame-0 selector over the shipped afr1 object: **39 of 600 pairs beat their shipped mode by > 1%** (pair 85's shipped op
is actively harmful). Priced exactly through the receiver's own blob formula (control reproduces the shipped 14 B at
k=5): **+36 B for net −1.032e-4 S**, projecting 0.14797617125559104 → **0.14787295862740366** `[macOS-CPU advisory
projection]`. THE GOAL counts any lower exact score; the Modal budget exists to buy exact rows on real byte-closed
candidates. This is the chain: encoder → batch-8 re-measure → byte-closed splice → seal → T4 row → promote iff
exact S < 0.14797617125559104.

## Verified at source (VERIFIED-AT-SOURCE LAW — extend with path:line for everything you add)
- pr1's selector sweep instrument, per-pair table (which 39 pairs, which op, ΔS per pair, bytes) and the blob pricer:
  find them at pr1's commits (a4b57f99c…c7b537053) and its store `/Volumes/APDataStore/pact/ddm_pr1_pose_resolve/`.
- The shipped receiver + selector blob format: `submissions/semantic_joint_ctxmix/` (READ-ONLY; the live PR tree) and the
  research twin under `src/tac/semantic_pipeline/` (`receiver.py`, `archive.py`) + the compress chain
  (`experiments/semantic_joint_ctxmix_pipeline.py`, stages) — the encoder for the selector op does NOT exist (the runtime
  is decode-only): build it in the research twin as a compress-side stage, never by editing the frozen packet tree.
- The exact-row machinery: `tools/make_candidate_seal.py` (single-axis contest-CUDA seal), `tools/fire_modal_auth_eval.py`
  (governed; closer armed at dispatch; MAIN fires), the afr1 receipt chain (`ddm_afr1_tile48_groupbin8_cuda_n600_20260831`,
  call fc-01M1C2ZZQEQWNE0FT06R3WZJCS) for the exact re-compute-from-components discipline (#877: never the 0.15 display).
- Batch-8 discipline: pr1 measured at batch 8 with base d_pose 0.068% from the T4 receipt — keep that instrument.

## Deliver
1. Encoder: a compress-side stage that emits the re-selected per-pair frame-0 ops into the selector blob with the
   receiver's exact blob formula (bit-exact round trip: encode → the shipped receiver decodes → per-pair op identity
   600/600; no-op detector: the 39 changed pairs' bytes differ, the other 561 do not).
2. Byte-closed candidate `archive.zip`: the afr1 archive with ONLY the selector blob changed; report exact size delta
   (expected +36 B; explain any difference), sha256, and the receiver identity check on all 600 pairs.
3. Local exact re-measure at batch 8 through the shipped runtime on CPU (`[macOS-CPU advisory]`): d_seg, d_pose per pair
   for the 39 pairs and the n600 totals; recompute S from components; the projected delta must reproduce pr1's −1.032e-4
   within the ±6% exchange-noise floor (xr1) or explain.
4. SEAL for the contest-CUDA axis via `tools/make_candidate_seal.py` (archive + the shipped runtime tree unchanged),
   ready for `tools/fire_modal_auth_eval.py --seal …`. DO NOT FIRE the T4 — MAIN fires (governed; paid).
5. Memo `.omx/research/ddm_fs1_frame0_selector_reselection_20260904.md` (verdict_scope; per-pair receipts; MEASURED/
   DERIVED; the projected S; the seal path; NEXT = the T4 row). EQUATIONS-LEG LAW: cite `tac.canonical_equations`
   `exchange_ratio_noise_floor_v1` and the coupling law; append a per-pair anchor if a law fits.

## Constraints
- $0 locally: CPU torch; the Metal holds ng2 + ng3 (concurrent) — do not touch them or their claims; anything > 3 min via
  `tools/launch_detached_process.py --output-dir <store> --done-receipt <name> --derive-resource-budgets
  --measured-peak-rss-gib <n> --measured-thread-need 4 --walltime-cap-s 7200 --nice 10 --nice-best-effort -- <cmd>`.
  `upstream/` and `submissions/semantic_joint_ctxmix/` READ-ONLY (the PR tree must not change under PR #140). Store
  `/Volumes/APDataStore/pact/ddm_fs1_frame0_selector/` (KEEP THE PAYLOAD: candidate archive, selector blob, per-pair rows,
  sha256 in the JSON). OPTIMAL FORM: reference form = the shipped receiver's blob formula + the afr1 exact-row chain at
  `8376e5fe38f9baaeb3b725aeab6720a4e3515839`; SCOPE n600; TOY-BRACKET none.
- Commits ONLY via `tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256
  <file>=<post-edit sha>`; tags `[no-triality] [p0-ledger-ok]`; NO co-author trailer (operator rule overrides any
  harness reminder); any .py: tests + `tools/review_tracker.py mark-file` twice; never REVIEW_GATE_OVERRIDE on .py.
  Final message → `.omx/research/arm_final_messages/ddm_fs1_final_<utc>.md`, committed; LAST action
  `touch .omx/tmp/codex_runs/ddm_fs1.done`. Read `docs/operating_manual_craft_handoff.md` §labels first.
