# ddm_sj1 pass 6 — Lagrange seg subset on the move 42 field (charter, MAIN 2026-09-10)

## Why
Pointer move 42 (rp1 round 2, S 0.1374765052591843 @ 180,238 B) moved on pose (4.66e-06) with seg
EXACTLY unchanged (d_seg 0.00010637 = 100·d_seg 0.010637 = 7.7 % of S). The seg-debt pool is the
largest remaining lever on this object: sj1's passes 3–5 (moves 32, 38) took −4.3e-3 / −8.1e-4 /
−1.1e-4 by token pre-distortion + carrier re-solve on Lagrange subsets, and the residual is renderer
boundary jitter at correct tokens (86 %, one pixel, Lane 40×). Pass 6 targets that residual on the
move 42 field, composed by re-verification (seg is sub-additive by pair overlap; rate anti-synergy
+6.4 %/token — memory `composition_of_disjoint_token_edits_is_subadditive_on_seg_by_pair_overlap_20260910`).

## Resume surface (read first)
- The move 42 packet `.omx/research/ddm_rp1_round2_rate_directed_predistortion_k192_20260910_pointer_move_42_20260910.md`,
  its seal, and `ROUND2_BASE.json` + `field_rebased` (sha aae528e6…) under `/Volumes/VertigoDataTier/pact/ddm_rp1_round2/`
  — the shipping field is move 42's archive; read the field from the SHIPPED bytes (parse-back), never from
  a sizing plane (`ddm_rp1_round2/n600/shard*/field_rp1_sizing.npz` is FORBIDDEN as a base: pass-4 body + edits).
- Your own passes: `.omx/research/ddm_sj1_multipass_token_predistortion_*`, the pass-5 packet (move 38),
  `experiments/ddm_sj1_pass5_price.py`, the admission chain (three-leg: seg on the shipped-mode decode,
  RESOLVED pose, real-encode rate), the subset-writer silent-revert fix (42d5fc651) and the stage-tail
  successor baseline rule (e6141fa4e).
- Laws: `pose_resolve_is_mandatory_after_every_field_change_20260910` (admit only on the RESOLVED pose; frame 0 =
  repair INSIDE the admission, never standalone — operator 2026-09-10 "Remember pose re solve" / "And frame 0"),
  `a_flag_and_a_constant_can_disagree_silently_guard_both_objects_20260910` (bind base field + ranking by sha),
  `first_order_token_price_is_a_ranking_never_a_charge_tail_flag_mass_untouchable_20260909`,
  `residual_seg_debt_is_renderer_boundary_jitter_at_correct_tokens_20260908`.
- Seal contract: `tools/make_candidate_seal.py` requires a decode-wall-clock leg; your receiver stays
  byte-identical to move 42's except archive.zip and the two pins, so inherit move 42's leg:
  `--inherit-decode-wall-clock /Volumes/VertigoDataTier/pact/ddm_rp1_round2/SEAL_ddm_rp1_round2_contest_cuda.json.decode_wall_clock.json`
  (if that sidecar does not exist, the move 40 sidecar `…/ddm_sj1_compose39_price/SEAL_ddm_sj1_compose39_rp1_union_contest_cuda.json.decode_wall_clock.json`
  is the `t4_direct` source; inheritance requires the pointer archive sha at seal time = move 42's — read
  `tac.decode_wall_clock.inherit_decode_wall_clock` and report if the pointer/source pairing refuses).

## Deliverable
Pass 6 on the move 42 field: rank the residual boundary-jitter cells, admit a Lagrange subset under the
three legs on the RESOLVED pose with frame-0 repair inside the admission, twin-price by real encode against
move 42's tail (never the ledger sum), pre-register falsifiers, seal. MAIN fires; packet move 43 iff exact
S < 0.1374765052591843. Report the seg/pose/rate decomposition, the subset size, and the yield curve vs the
bar. If the pool is exhausted at formulation scope, say so with verdict_scope and the measured curve.

## Boundaries
- CLAUDE.md non-negotiables; commits ONLY via `tools/subagent_commit_serializer.py … [no-triality] [p0-ledger-ok]`
  with post-edit shas (zsh arrays); `.py` two review passes; no co-author trailer; never edit `upstream/`,
  the PR tree, or a sealed tree; keep every payload (SSD tier, ≤ 8 GiB per store, hardlink identical raws
  with certificates); launch heavy steps through `tools/launch_detached_process.py --done-receipt …`.
- Host: the rlc2 codex arm (light) and vr8 (custody audit; may hash GBs) share the host; no timing windows
  are planned — commits are fine.
- Checkpoint as `ddm_sj1` (step numbering continues from your last) every ~10 tool uses.

Final message: what you MEASURED, what you did NOT, every boundary, the seal path or the typed blocker, and
the frontier line `composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)` — a
new number only if YOUR measurement moved it.
