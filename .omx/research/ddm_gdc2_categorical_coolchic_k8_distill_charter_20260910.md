# ddm_gdc2 — categorical Cool-Chic distillation of the retained K=8 scanline teacher: build the trainer, run the governed burn, close the packet (charter, MAIN 2026-09-10; MAIN GO GRANTED for this burn under the operator's standing GO for frontier-lowering burns, 2026-09-02)

## Why this burn
Pointer move 43: S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600] (d_seg 0.00010345, d_pose 4.59e-06). Sub-0.12 is
the RATE corner: at move 43's held distortion the archive cap is **154,507 B**, demand **−25,959 B**; the replaceable token
tail is 119,969 B (66.48 % of the archive). gdc1 (`.omx/research/ddm_gdc1_generator_door_construction_design_20260910.md`, sha c924bb6a831e7a8f…, commit 01b13a69b) derived the success inequality
**`P_program + R_exact ≤ 94,010 B`** (program packet + the REAL coded exact residual back to the move-43 field, so the
receiver renders the identical token field and the shipped score is unchanged) and measured the ordered scanline program
FORMULATION-NO-GO (best K=6 → 329,122 B). It retained a full-n600 **K=8 teacher render** (packet 306,042 B; 88,304
mismatches vs the move-43 field, Lane-concentrated) and wrote the governed spec for the surviving construction: distill
that teacher into a counted categorical coordinate decoder (Cool-Chic form: int8 latents + tiny depthwise CNN + 5-logit
integer head) that must reach **≤ 68,322 B at exact teacher identity** (4.4794×), after which the authority gate is
`packet + best real exact residual ≤ 94,010 B`, receiver parse-back exact, deterministic repeat, full-n600 decode ≤ 900 s.

## Your deliverable — gdc1 §"Governed GDC2 launch spec" implemented VERBATIM (read it first; lines ~232–268)
1. Build the trainer + receiver in `experiments/ddm_gdc2_categorical_coolchic_distill.py` (+ a NumPy-fp32/integer receiver
   module under `src/tac/` if reusable): fixed architecture exactly as specified (z0[75,24,32,4] int8, z1[150,12,16,2] int8,
   trilinear coordinate interpolation, five width-24 depthwise-separable 3×3 blocks, five-logit integer head; all latents,
   weights, biases, scales COUNTED); MLX training = research signal, NumPy/integer receiver = verdict authority with
   MLX/NumPy logit parity + exact argmax identity checked at every stage; seed 20260910 everywhere; deterministic tile
   schedule generated once and retained; EMA 0.999 exported as the authority at every stage.
2. Stages A (10,000 AdamW steps, lr 3e-3→3e-4 cosine, 16 seeded blocks of 8 pairs × 64×64 sites/step, categorical CE
   vs the K=8 teacher) → B (8,000 steps, lr 1e-3→1e-4, + measured symbol-rate proxy; λ ∈ {1e-4, 3e-4, 1e-3} as three
   separately RETAINED branches) → C (4,000 QAT steps, lr 1e-4, integer receiver in the loop). Atomic checkpoints
   (optimizer/RNG/EMA/config) every 250 steps; distinct stage-end + branch checkpoints never overwritten; `--resume-from`
   exact continuation. Early stop rule exactly as specified (Pareto-dominated at two consecutive checkpoints, or divergence).
3. At every stage end: raw latents/weights, all three coded packets (Brotli-q11 / zlib-9 / LZMA2-extreme) + repeats,
   full-n600 render, mismatch count vs the move-43 field AND vs the teacher, hashes, decoder timing — all retained under
   `/Volumes/VertigoDataTier/pact/ddm_gdc2_categorical_coolchic_k8_distill/` (storage waterfall + 40 GiB reserve;
   certify-or-block; no deletion; hardlink byte-identical repeats with certificates).
4. Closure: for the best branch, code the REAL exact residual back to the move-43 field with the real coder
   (gdc1's `experiments/ddm_gdc1_scanline_partition_program.py` residual path; quote the measured B/mismatch, not
   gf1's 0.2909), report `packet + R_exact` against 94,010 B, full-n600 decode wall-clock, and parse-back identity.
   NO scorer, NO Modal, NO candidate archive: if the gate passes, STOP and hand MAIN the exact numbers; MAIN allocates
   the scorer step.
5. Launch discipline: `tools/claim_lane_dispatch.py claim --lane-id ddm_gdc2_categorical_coolchic_k8_distill_20260910 …`
   first; every burn through `tools/launch_detached_process.py --output-dir <Vertigo run dir> --nice 0 --done-receipt
   <name> --derive-resource-budgets -- <cmd>`; `tools/witness_memory_preflight.py`-class projection of peak RSS before
   the first burn (shared host: rlc5 is live; keep ≤ 40 GiB); heartbeat/telemetry rows per 250 steps (full telemetry
   always); no MPS score claims; `research_only=true`, `score_claim=false` on every row.
6. Memo `.omx/research/ddm_gdc2_categorical_coolchic_k8_distill_20260910.md` (stage table: steps, packet bytes per coder,
   mismatches vs teacher / vs field, parity, timing; the closure arithmetic; every boundary) — commit early and often via the
   serializer (two review passes per .py; `[no-triality] [p0-ledger-ok]`; no co-author trailer). Checkpoint as `ddm_gdc2`
   every ~10 tool uses. Long burns: end your turn with the launch receipts and the resume command; MAIN resumes you.

## Boundaries
Never edit `upstream/`, the PR tree, sealed trees, or live arm directories (`ddm_rlc5_cure_on_move43`, `ddm_sj1_pass6*`);
read-only on the move-43 field and gdc1's retained store; no Modal, no scorer, no candidate archive; no unrecorded
autotuning; no toy subsets (n600 renders; the 8-pair × 64×64 training blocks are the SPEC's schedule, not a scope cut).

## OPTIMAL FORM
- Reference form: gdc1's governed spec verbatim + the official Cool-Chic implementation (Orange-OpenSource/Cool-Chic) and
  paper as mechanism anchors; the K=8 teacher render as the exact target. A reduced STEP COUNT for a first parity smoke is
  SCOPE (declare it; it produces no verdict); any change to architecture/latent sizes/loss is MECHANISM (toy-bracketed).
- Provenance pins: gdc1 memo sha c924bb6a831e7a8f…; gdc1 RESULT `.omx/research/ddm_gdc1_20260910/RESULT.json` (record sha); K=8 teacher
  store `/Volumes/VertigoDataTier/pact/ddm_gdc1_generator_door/scanline_v1/` (record the render's path/bytes/sha); move-43
  field sha 78e57545… (record full); pointer commit 48109233e / archive 7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e.

## Prior negatives accounted (operator 2026-08-15)
- gdc1: scanline program closed at formulation scope; gf1's 0.2909 B/mismatch is not a successor rate (K=6 measured 0.4508).
- gf1: form and fit are one fact — the distilled decoder's expressive ceiling is measured by its own mismatch count.
- md1–md4: born-vehicle accuracy is data-anchored at Lane edges; the teacher's mismatches are Lane-concentrated — report
  the Lane share of residual bytes separately.
- bz2d: token error ×1.157 to argmax — irrelevant here ONLY because the residual restores the field exactly; if any branch
  ships a non-exact field, it is a different (seg-paying) object and must say so.
- ng4 / sd1 / #205: stage entry restarts the objective; read losses at fixed τ; project peak RSS before launching.

Final message: stage table, best-branch packet bytes + mismatches, `packet + R_exact` vs 94,010 B, decode wall-clock,
retained paths, the resume command if unfinished, every boundary, and the frontier line
`composition S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600] (move 43)` unchanged.

<!-- # FORMALIZATION_PENDING: burn charter; the closure arithmetic lands in the resulting memo's equations leg -->
