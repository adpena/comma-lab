# ddm_wc1 — WALL-CLOCK optimal form for the lifted trainer (prep now, bench at the Metal gap)

**Operator steer (2026-08-07):** wall-clock optimizations dangling under our noses. The named
one: `experiments/ddm_mx1_pr130_semantic_renderer.py` + `src/tac/pr130_lift/mlx_semantic_renderer.py`
were lifted for FIDELITY (A/B discriminator symmetry), never throughput-audited. The symmetry
constraint binds ONLY the two n32 arms — n120/n600 inherit whatever trainer exists at fire
time, and both future arms inherit optimizations IDENTICALLY (no confound). Current pace
~7.85 s/step at n32; naive n120×6000 projects ~2 days. Target: 3-6× via banked, measured levers.

**Doctrine:** wall-clock is a JOINT objective (m32); bit-identity binds DECODE only — the
training gradient path may take faster numerics. MIN-WALL-CLOCK d_seg = P0 (m33). FREE smoke
first. Constants-are-poison: derive, never copy.

**Recall-first — the banked speed levers (receipts exist, do NOT re-derive):**
- 1-thread training pin: 2.96× measured on our line (config †D discipline). Check what the
  lifted trainer/launch env actually sets (OMP/MKL/veclib threads, mx threads).
- Batched pair forward: #261/#313/#447 (the 2-4× step lever; witness-line bit-identity twin
  precedent). The guard's microbatch_pairs key + 21 GiB projection vs ~57GB free = headroom.
- mx.compile / fused regions: #356/#357 megakernel + safe-compile lessons (which regions are
  compile-safe, which broke).
- fp16/bf16 in the TRAINING path only (never the decode/byte-close path): #496/#509 lessons.
- Async/subprocess verdict reclaim (#330) + pose-verdict gate (#495) — verdict-side wall-clock.

**Deliverables:**
1. THROUGHPUT AUDIT of the lifted trainer: for each lever {present?, applicable to this conv
   vehicle?, expected multiplier w/ receipt, risk to gradient fidelity}. Honest N/A where the
   architecture makes a witness-line lever inapplicable (e.g. grouped-backward coord-INR kernels).
2. BUILD the applicable levers behind flags, DEFAULT-OFF, byte-identical-off (verify: same argv
   without flags → identical behavior; add a fast CPU test per lever). No live-run mutation;
   the n32 A/B stays untouched.
3. A ONE-SHOT BENCH SCRIPT (≤10 min Metal budget) MAIN fires in the natural Metal gap between
   ARM-CAP endpoint (~22:40) and the ARM-VEH fire: measures s/step for {baseline, +threads,
   +batched, +compile, +fp16-train} on a short n32 window from the live checkpoint, emits a
   receipts JSONL. Include a d_seg-batch sanity column so a numerics-degrading lever is caught
   at the bench, not at n120. THE ARM DOES NOT FIRE METAL ITSELF (ONE-Metal-fire law; ARM-CAP
   owns it) — prep + CPU verification only.
4. DERIVED n120 step count: from mx1t's facet receipts (margin-trend/churn curves + tail-average
   policy), derive a step-count recommendation with the marginal-d_seg-per-1000-steps table —
   labeled DERIVED w/ the measurement it came from, never a copied 6000.

**Boundaries:** CPU-only in-arm, NO Metal, NO scorer slot, read-only toward live run dirs.
Findings: `.omx/research/ddm_wc1_20260807/WC1_FINDINGS.md` + bench script + receipts schema.

**Discipline:** serializer + POST-EDIT `--expected-content-sha256` per file; tags
`[no-triality] [p0-ledger-ok]`; review_tracker ×2 per .py; NO Claude/AI attribution or
Co-Authored-By trailer. If serializer hits sandbox git-perms, write artifacts + say so.
