# L5 v2 subagent risk review: Volta

Recorded: 2026-05-17T05:56:58Z

Subagent: `019e3463-b4f4-7b13-b47b-d4f2fdb5bd72` (`Volta the 3rd`)

Mode: read-only review. No files were edited by the subagent. This memo preserves
the review signal so it is not trapped in chat-only state.

## Findings

1. TT5L side-info usefulness remains unproven at full trained shape.

   The full 600-pair recovered TT5L packet has `0/27000` nonzero side-info
   values. Current code can emit nonzero side-info in a 2-pair
   `[macOS-CPU advisory]` smoke, but no full 600-pair trained archive proves
   useful nonzero side-info yet.

2. Tracked custody artifacts lag current no-op hardening.

   Current `src/tac/optimization/tt5l_sideinfo_variant_packets.py` records
   source-to-variant archive/member/side-section change booleans, but tracked
   variant packet JSONs lack those fields. Regenerate the durable `.omx/research`
   artifacts before using them for dispatch or architecture-lock evidence.

3. Provider and axis schemas are easy to misfeed.

   Result review packets use `score_axis`, while effect-curve cells require
   `axis` plus `variant`. Seed-cell wrappers normalize one path, but direct
   reviewed-result ingestion can still produce missing-axis or missing-variant
   failures. Pair identity must remain validator-enforced rather than inferred.

4. Architecture lock remains correctly blocked.

   `.omx/research/l5_v2_architecture_lock_packet_20260516_codex.json` still has
   `architecture_lock_allowed=false`. Current blockers include missing valid
   side-info effect curve, missing first-anchor timing smoke, missing
   paired-axis/probe evidence, Modal billing-limit blocker, and Lightning
   alternate-provider prerequisites.

## Next actions

- Produce a full 600-pair current-code TT5L trained archive with nonzero side-info.
- Rebuild zero, random_lsb, shuffled, trained, and ablated variants from that source.
- Run claimed paired `[contest-CPU]` and `[contest-CUDA]` exact eval for all five cells.
- Regenerate tracked variant, effect-curve, and materialized-work-unit artifacts from current code before any new dispatch or architecture-lock refresh.
