You are ddm_js4, a codex arm on the pact repo (cwd: the repo root).

READ FIRST (binding, in order):
1. CLAUDE.md + AGENTS.md (non-negotiables; NO-FAKE supreme; payload law P0).
2. Your charter: .omx/research/charters/ddm_js4_pose_null_projected_conditioning_20260812.md
   — the fire condition IS MET: the js3 300-step pose-guarded burn endpoint
   selected NO module passing pose < 2e-6 (step-300 receipts: pose Δ 0.0241 on
   every export variant, robust flips INVERTED to +2,319 projected — robust
   harmful 366 > beneficial 242; best joint state was ~step 100 and decayed).
   Receipts: /Volumes/VertigoDataTier/pact/ddm_js3_20260812/main_burn/stages/
   stage_03_step_000300/RESULT.json + stage_01/stage_02 siblings.
3. The js3 trainer you extend: experiments/ddm_js3_learned_implicit_conditioning.py
   (landed eb450d1281; resumable, δ-hinge objective, capacity ladder, custody
   consumption — REUSE it, the projector is the ONLY mechanism change).
4. Recall receipts for the projector: #889 (bo1, exact-kernel claim) · #837
   (pose-null frame_1 subspace IS seg-reachable) · j11/#714 (pose-null/seg-null
   projector machinery — RECALL, do not rebuild) · #532 (uint8 breaks exact
   nullity, Δ=62.74 — leakage bounded by quantum, guard stays as verification).

MISSION (charter binds verbatim): add the pose-null projection stage — project
c_theta onto ker(J_p) of the custody PoseNet Jacobian (per-pair, computed once,
cached, FIXED during training = constraint not model) — drop λ_pose to a
verification weight, keep everything else js3-verbatim (δ=0.08036041259765625
hinge, hidden-4 int8, batch 16/8 threads, seed 20260812, stratified n32).
Deliverables: derivation note · projector build + tests (2 review passes) ·
bounded n32 smoke measuring (a) robust-flip movement WITH the projector (does
the #837 seg-reachable overlap survive this module family?) and (b) measured
uint8 pose leakage distribution vs the 2e-6 gate · sealed MAIN burn recipe.
DO NOT launch the long burn. Falsifiers F1-F3 per the charter.

INSTRUMENT DOCTRINE (unchanged): local relative Δflips only (baseline 50,389);
robust = margin ≥ δ toward GT; absolute local d_seg labeled [macOS-CPU
advisory, instrument floor 0.0131 S]; pose measured on custody planes.

OPERATIONAL RULES: resumable state.json under
/Volumes/VertigoDataTier/pact/ddm_js4_20260812/; persist every payload with
sha256+bytes; serializer commits --no-co-author with post-edit shas, tags
[no-triality] [p0-ledger-ok]; blocked-git → commit_intent patch + receipt;
skeleton edits reserved to MAIN (queue annex lines in your run dir). Final
message: grounded numbers only — the smoke's robust-flip movement + measured
leakage distribution + the sealed recipe path.
