# ddm_eq1 — the equations leg, backfilled: 29 memos, one new law, and the detector put back on the path

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-04 · Cost: $0

## Why (triality)
A finding is KNOWN only when DAG, DSL and equations agree. Catalog #344
(`check_empirical_finding_memo_references_canonical_equation`) measures **29** `.omx/research` memos dated
2026-08-27..09-03 that state empirical findings with NO canonical-equation reference and NO waiver — a week of arm
output drifted past the equations leg. Two of the week's strongest measured laws have no equation at all.

## Verified at source (VERIFIED-AT-SOURCE LAW)
- Gate: `src/tac/preflight.py` `check_empirical_finding_memo_references_canonical_equation` — reference tokens
  `_CHECK_344_CANONICAL_EQUATION_REFERENCE_TOKENS` at :85204 (`tac.canonical_equations`,
  `canonical_equations_registry`, `register_canonical_equation`, `update_equation_with_empirical_anchor`, …);
  waiver marker `FORMALIZATION_PENDING` (:85215). Wired `strict=True` in `preflight_all` at :7510 — yet 29 live.
  **First question, answered at source:** why does the commit-time hook (`tools/preflight_hook.py`) pass with 29
  live? (subset of checks? scan window? date filter?) Record the answer; if the gate is simply not on the commit
  path, put it there in the SAME batch that drives the count to 0 (strict-flip atomicity), or record why not.
- Live list (29): run the gate with `strict=False` and take ITS list — do not hand-enumerate. Pinned at commit
  `d3212bed117e1124952333e9b1c0909a9b7feef6`.
- Registry helpers: `tac.canonical_equations.registry.register_canonical_equation`,
  `update_equation_with_empirical_anchor`, `update_equation_with_domain_refinement` (append-only events; the
  latest event per equation_id is the current state). Example of a full module: `src/tac/canonical_equations/
  margin_band_satisficing_threshold_20260712.py`; example of a registration tool:
  `tools/register_aa_sdf_observation_render_equation.py`. Registry `.omx/state/canonical_equations_registry.jsonl`
  is TRACKED — commit it with the module.

## Deliver, in this order
1. **Register the coupling law** `renderer_seg_pose_coupling_shipped_object_v1`: |Δd_pose|/|Δd_seg| on the shipped
   afr1 renderer, two independent anchors — rf1 166.8 (structural change,
   `.omx/research/ddm_rf1_renderer_film_rung_20260824.md`) and ft1 217.30 (trained change,
   `.omx/research/ddm_ft1_shipped_renderer_aligned_finetune_20260903.md`, `retained/verdict_ft1_step600.json` under
   `/Volumes/VertigoDataTier/pact/ddm_ft1_shipped_renderer_aligned_finetune/`). Domain: shipped SM3R renderer at its
   own size, seg-only gradient direction; excluded: joint (pose-priced) formulations. Consumer: the fold-back program
   (`ddm_fb1`) and any future renderer charter (the derived closing arithmetic: a seg cut of fraction f costs
   Δd_pose ≈ coupling·f·d_seg_base; payable ceiling 1.694e-5 same-object). Memory to cite:
   `renderer_seg_pose_coupling_170_220_two_arms_20260903`.
2. **Register the prefix-bias detector law** ONLY IF no equation already covers it (check `tools/list_canonical_equations.py
   --json` for prefix/cohort laws first — [[m88]]/bp2 may already have one; if so, append dr1's n600 δ_R anchor to
   THAT law instead): annulus-restricted statistics of a contiguous prefix are biased even when the global-pixel
   statistic is not (dr1: annulus p95 +11.70%, global +0.45%).
3. **Sweep the 29 memos** (append-only: add a dated addendum line, never rewrite bodies; Catalog #110/#113): for
   each, EITHER cite the equation it anchors/refines (with the literal token `tac.canonical_equations` and the id;
   append an anchor via the helper when the memo carries a real measured row not yet anchored) OR a same-line
   `# FORMALIZATION_PENDING:<substantive rationale naming the law it would need and why it is not yet derivable>`.
   Review/process memos (fr2, pq13, ht1, hv3, hv4, rc_precheck, hp1, fpr1, ql2) are waiver cases with an honest
   one-line rationale; measured-verdict memos (qbt2b r3–r10, qbt1, gf2, ar1, ft1, fcd1, lc3, jc1, na11, nx1, qbr1,
   qn1, SPEC_qbflow) must name a law (existing or newly registered) — no blanket waivers on measured rows.
4. **Drive the gate to 0 strict**, make `src/tac/tests/test_check_344_canonical_equation_referenced.py` pass
   (`test_live_repo_regression_guard` + `test_wave_3_backfill_keeps_live_count_zero_in_strict_mode`), and answer
   the commit-hook question from the "verified at source" block.
5. Memo `.omx/research/ddm_eq1_equations_leg_backfill_20260904.md`: table (memo → equation or waiver rationale),
   the hook answer, GESTALT-DELTA line, NEXT_IF_RESUMED. Final message → `.omx/research/arm_final_messages/
   ddm_eq1_final_<utc>.md`, committed. LAST action: `touch .omx/tmp/codex_runs/ddm_eq1.done`.

## Constraints
- $0; no scorer, no Metal, no Modal; the QBR1 burn is live — never touch its custody. `upstream/` and
  `submissions/semantic_joint_ctxmix/` read-only. No /tmp paths.
- Commits ONLY via `tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256
  <file>=<post-edit sha>`; tags `[no-triality] [p0-ledger-ok]`; NO co-author trailer (operator rule overrides any
  harness reminder). Any .py: tests + `tools/review_tracker.py mark-file` twice with a real second read; never
  REVIEW_GATE_OVERRIDE on .py. Read `docs/operating_manual_craft_handoff.md` §labels first.
- OPTIMAL FORM: reference form = the canonical registry helpers and the existing equation-module pattern at commit
  `d3212bed117e1124952333e9b1c0909a9b7feef6`; SCOPE = the 29 live memos + 1–2 laws; no mechanism reduction; TOY-BRACKET: none.
