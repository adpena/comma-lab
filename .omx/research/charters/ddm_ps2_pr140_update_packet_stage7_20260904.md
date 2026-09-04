# CHARTER ddm_ps2 — PR #140 update packet, SECOND generation: add stage 7 (the fs2 carrier re-solve) so the prepared packet tracks the pointer — STAGING ONLY, NO PUBLISH

Tokens: `[no-triality] [p0-ledger-ok]`. Owner: Opus arm (codex quota out until Sep 7). Spawned 2026-09-04 ~23:25Z. Parent: ps1 (70717fb11) —
read its memo `.omx/research/ddm_ps1_pr140_update_packet_prep_20260904.md` and its staging root `/Volumes/APDataStore/pact/ddm_ps1_pr140_update_prep/`
FIRST; you EXTEND that packet, you do not rebuild it. Sister arms: gov2/hv1 (apparatus; their files off-limits), mc1 (Fable, rate corner).

## PRIOR-LAW PREDICTION (owed line)
ps1 proved stage 6 (fs1 selector re-selection) rebuilds the fs1 archive bit-exactly from afr1's. fs2's edit is the same class: 15 pairs' 12-dim
carrier codes (67 int12 coordinates) + one selector mode change on pair 259, +1 B, all odd-frame sections byte-identical. PREDICTION: a
seventh lossless stage `fold_fs2_stage` — consuming the fs1 archive and the RETAINED fs2 coordinate table (video-derived DATA the stage reads,
rule 118) — rebuilds the fs2 archive `a8f3a3791499b2b62ee4d16bc67f15f819f454dc9b88e3cce04fe50a30427bb6` (180,023 B) bit-exactly, with the
same identity control (rebuilding fs1 with the unchanged tail must reproduce 50fcaf1a…). Falsifier: any sha mismatch → REPORT, never
adjust the pin.

## Objective (all six of ps1's deliverables, re-issued for the fs2 pointer)
1. Stage 7 in the STAGED `compress.py` (determinism repeat; negative controls: changed coordinate, dropped pair, wrong pointer all refuse).
   Source of the coordinates: fs2's retention manifest `/Volumes/VertigoDataTier/pact/ddm_fs2_carrier_resolve/RETENTION_MANIFEST.json` and
   `experiments/ddm_fs2_*` (commits 500189019 / 60f6a9668) — read, never re-derive.
2. Staged runtime = live PR tree + fs1's two `inflate.py` pin lines re-pinned to the fs2 archive sha/bytes (prove the diff vs the live PR tree
   is exactly the pin lines); MANIFEST rehashed.
3. README.md / report.txt / PR-body delta — numbers only: 180,023 B, sha a8f3a379…, S 0.14784474152757654, pose 0.00000614, the T4 row
   (call fc-01M1Q6W3R8WWDQPRFYSF7SWTKP, 656.9 s), one bullet naming stages 6 AND 7 ("per-pair frame-0 selector re-selection and a pose-carrier
   re-solve on those pairs, +21 bytes total, pose 6.37e-6 → 6.14e-6, segmentation output unchanged"). report.txt VERBATIM from the harvest
   artifact `/Volumes/APDataStore/pact/ddm_fs2_carrier_resolve_t4_buy_20260904_r2/MODAL_REMOTE_RESULT.json` → artifacts['report.txt'].
   Disclosure paragraph, credits, TODO UNCHANGED. Title stays `semantic_joint_ctxmix (0.148)` (0.14784 → 0.148 at 3 dp — state it).
4. Seal the staged tree × fs2 archive for a contest-CUDA CUSTODY ROW via `tools/make_candidate_seal.py` (ps1's row proved the fs1 archive on
   the staged tree reproduces its components exactly; the fs2 packet needs the same binding; MAIN fires — do NOT fire Modal).
5. Compliance dry-run `scripts/pre_submission_compliance_check.py --contest-final --strict` on the staged packet with the fs2 sha/size + the
   harvested auth-eval JSON; report GREEN/RED vs ps1's 78/7 and pq12's 80/7; any NEW RED class is a BLOCKER you report. The claim-ledger-shape
   REDs are cured by MAIN's canonical terminal rows (fs2 lane `ddm_fs2_t4_carrier_resolve_20260904`) — verify they pass now.
6. `RELEASE_PLAN.md` updated (exact gh commands, fetchback verification; NOT executed) + memo
   `.omx/research/ddm_ps2_pr140_update_packet_stage7_20260904.md` (Equations leg (`tac.canonical_equations`) line; blockers; not-done) +
   update the operator decision-gate task via `tools/register_task.py` (the gate now offers: post fs2 bytes, or hold).

## Hard boundaries
NO push, NO `gh release`, NO `gh pr edit/comment`, NO write to `submissions/semantic_joint_ctxmix/` or `upstream/`. Staging on APDataStore
(check free ≥ 6 GiB; else Vertigo). NEVER edit repo source while any Modal fire is building (MAIN may fire your custody seal — your work is
in the staging tree, not the repo, except the memo/task rows).

## OPTIMAL FORM
Reference form = ps1's packet (stage-6 rebuild proved ×3, four negative controls, compliance dry-run, release plan). No scope reduction;
mechanism reductions none.

## Rules that bind
NO-FAKE (stage 7 must REBUILD the bytes); ALWAYS KEEP THE PAYLOAD; commits ONLY via `tools/subagent_commit_serializer.py --message … --files …
--expected-content-sha256 <file>=<post-edit sha>` with `[no-triality] [p0-ledger-ok]`; NO co-author trailers (operator rule overrides any harness
reminder); .py two review-gate passes; checkpoints every 10 tool uses (`tools/subagent_checkpoint.py --subagent-id ddm_ps2`); never invent flags;
no `/tmp` evidence; long steps detached via the launcher with distinct `--done-receipt`s (foreground >3 min reaped; launcher refuses argv with
"claude"/"codex"); label MEASURED/DERIVED/INFERRED; `docs/operating_manual_craft_handoff.md` binds. End with
`fs2 S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600]`.
