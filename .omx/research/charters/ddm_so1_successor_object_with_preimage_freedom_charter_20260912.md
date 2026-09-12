# ddm_so1 — the successor object: byte feasibility WITH pre-image freedom (design + $0 first rung; charter, MAIN 2026-09-12; operator full-authority GO + "be creative and weird, think divergently"; codex astra xhigh)

## The problem, as measured (read these before thinking; never recall from memory)
- The live object (move 48: S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600]; token field 118,896 B, renderer member 29,862 B,
  HPAC prior 11,629 B, remainder 18,624 B; d_seg 0.00010345, d_pose 4.59e-6) is a measured conditional optimum of the joint problem on every
  axis: `.omx/research/ddm_gs3_gestalt_after_submission_20260903.md` Addenda 47–53 (read all seven); ren2 (`ddm_ren2_*_20260912.md`),
  rq1 (`ddm_rq1_*_20260912.md`), dpi1, tmx1 memos. The field is the renderer's PRE-IMAGE: rendering the true partition scores 2.825× worse
  than rendering the field (ren1 §3). Demand: −24,604 B at held distortion, or 23× less distortion at held bytes.
- THE CROSS (memory `the-cross-two-objects-each-hold-one-half-of-sub012`, canonical memo `ddm_qbt2b_r10_doubling_adjudication_20260829.md`
  + `ddm_xo1_MAIN_adjudication_20260830.md` §7): the born/generator object has the rate (bz2: 100,862 B = generator packet 47,779 +
  semantic_renderer 30,856 + pose_carrier 22,010 + residual 96) with distortion 0.33; the intersection {byte-feasible} ∩ {reachable
  distortion} is MEASURED EMPTY at n=3 (NR1 K32: 27.7 distortion at 122,250 B). "Small" does not predict distortion.
- Laws that bind any successor: pre-distortion is what buys seg (jg1: 95.9 % of seg debt is render→re-segment loss; sj1 multipass memo
  `ddm_sj1_multipass_token_predistortion_20260905.md`); the repair family exhausts per object and re-opens on re-render; token −log2p is a
  ranking not a charge; the tail's best receiver-visible oracle is 8,365 B short (`lane_surprise_atlas_*_20260911` memory); generator form
  is 2.18× cheaper than model+coded tokens (`generator-form-is-2x-cheaper-than-model-plus-coded-tokens`); accuracy half closed by the
  round-trip intercept; carrier lattice closed (pc2/pc3); Lane = 0.59 % area / 33.5 % bits / 40× over-represented in every damage class.

## Deliverable ($0; NO launches, NO Modal, NO training; read-only over sealed trees and other arms' dirs)
1. A DESIGN MEMO `.omx/research/ddm_so1_successor_object_with_preimage_freedom_20260912.md` that answers ONE question with derivations
   and receipts: what construction can hold BOTH halves — a byte-feasible representation of the 600 label planes (the born object's
   ~48 KB generator packet class, or anything else you can DERIVE to ≤ ~95 KB total with the shipped renderer/carrier) AND per-token
   pre-image freedom so a field can absorb the renderer's bias (the property that bought every move from 24 to 48)? Candidates you must
   price on paper with the store's measured constants (cite each): (a) generator + sparse residual field (residual = the pre-distortion
   edits only; price its entropy from the pass-1..6 edit counts and bits/token); (b) coarse token lattice + deterministic decode-time
   boundary regularizer (free compute inside the 30-min budget; ~700 s of T4 slack measured) + residual; (c) lane as an 8-dim
   generator with the rest of the field coded as today (price with ls1's atlas: lane 33.5 % of bits) + lane residual; (d) anything
   weirder you can DERIVE — but every candidate gets a byte estimate, a distortion mechanism, and the SINGLE falsifier that would kill it.
2. Rank by (derived bytes at derived distortion) → S, with an honest interval, and name the ONE $0 first rung that measures the ranking's
   binding assumption on the real object at n600 (e.g. "the residual after generator G on the current field has entropy X bits — measure it
   with the real coder"): exact command sketch against real tools (grep argparse; never invent flags), inputs by sha, expected number,
   falsification threshold. A rung whose result is not exact bytes or n600 argmax cells is not a rung.
3. Verdict scope on every negative; MEASURED / DERIVED / INFERRED / ASSUMED on every number; `# FORMALIZATION_PENDING:<rationale>` or a
   canonical-equation cite in the memo.

## Process (codex arm rules)
Serializer commits only (`tools/subagent_commit_serializer.py --message "… [no-triality] [p0-ledger-ok]" --files … --expected-content-sha256
<file>=<post-edit sha>`; `REVIEW_GATE_OVERRIDE=1` for the .md); NO co-author trailer, NO AI attribution. If the serializer refuses with a
Git-object write denial (rc 17) this is NOT a stop: keep the file in the working tree, leave the verified bundle, and report; MAIN lands.
Commit LAST, once. Checkpoint as `ddm_so1` via `tools/subagent_checkpoint.py`. Read `docs/operating_manual_craft_handoff.md`. Never edit
`upstream/`, `submissions/semantic_joint_ctxmix/`, sealed trees, contract code. Do not touch `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7`
(a live Opus arm).

## OPTIMAL FORM
Reference form: the cross memos + the sj1 multipass memo + the born-object memos (`ddm_qbt2b_*`, `ddm_bz2*`, `ddm_xo1*`) read at source;
every constant cited with its receipt; no number from memory. Prior negatives: the cross (n=3 empty intersection); md1 (62 % of
persistent partition error unreachable); mc1 (motion-compensated plane closed); bd1 (temporal saturated); ls1/ls2 (tail oracle short);
ren2/rq1 (renderer axes closed both ways); fb1 (no single axis perfected reaches sub-0.12). A design that re-proposes any of these
without a NEW mechanism is not a candidate.

Final message: the ranked table, the one first rung with its exact command and falsifier, the memo path + sha, commit rc, and the line
`composition S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600] (move 48)` unchanged.

<!-- # FORMALIZATION_PENDING: charter, not a finding; the arm memo carries the cite -->
