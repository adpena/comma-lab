---
schema: dag_feed.v1
feed_id: FEED-bc1-qa24-compose-fire
date_utc: 2026-07-30
arm: ddm_bc1
pointer: "0.1910828242 [contest-CPU] UNMOVED"
score_claim: false
tokens: [p0-ledger-ok]
---

# FEED-bc1 — QA24 5-piece composed seg re-burn: BUILT + FIRED (§3.5 measured-reformulated)

- **QA24 FIRED** (governed detached, standing GO). pid 68621 · ticket_hash 81e9f26c239bcc5c ·
  git e0a37e82f4 · out `/Volumes/VertigoDataTier/pact/ddm_bc1_20260731/burn_out` · variant=lotto
  D16/c4 w24 400ep/480min · Metal grouped-backward ACTIVE · solve_project init ✓ · ep0 ep_loss 60.3.
  Endpoint = MAIN's post-burn re-solve charter. Pointer moves ONLY through a byte-closed eval row (none yet).
- **5 pieces BUILT + tested + committed** (1dcc71d2d5 · c2082f701d · e0a37e82f4):
  §3.1 coarse-grid cell-mask (token-zero + gradient-vanish + byte-close exclusion) · §3.2 margin-weight ·
  §3.3 lattice-anneal STE@knee · §3.4 rate-in-loss soft-entropy · §3.5 = directional-delta. 11 unit tests.
- **§3.5 MEASURED-REFORMULATION (MAIN Option A GO):** the absolute bounded stage-exit pose solve is
  INSTANCE-DEAD on this vehicle — 4 solvers (Adam ~10 / FD-GN ~30 / analytic-STE-GN ~29 / p3v2-cosine6
  ~11-16) all plateau d_pose ~10-38 vs post-burn ~0.0016; diagnostic GT_f0+GT_f1 → 9.7e-12 (target
  reachable, single-seg-frame recovery is not). ADOPTED = DEGRADED directional-delta
  d_pose(GT_f0,burn_f1)−baseline (VALIDATED: noise floor EXACTLY 0.0; sky/hood-frozen delta 0.14215 =
  knee_sensitivity). LITE-absolute INSTANCE-DEAD · LITE-delta ADOPTED · FULL bilevel (v6) unaffected.
- **QA79 (bicubic) settled:** burn R up-lift ALREADY bicubic (train-R + decode-R consistent, anti-Gibbs);
  race CONFIRMS bicubic wins d_seg 0.000960 vs bilinear 0.001044 (−8.4e-5). Only other our-code bilinear =
  the warp = post-burn pose decode (d_seg-invariant), deferred. Zero counted bytes; dimension closed.
- **DSL:** 5 QA24 levers + `qa24_composed_burn_program` in `spec_tr1_renderer_20260728` (SoT). Launcher
  gate0 (venv custody) landed. QA66 pose-tail `--composed-s-subset-ids` hook + `--composed-s-delta-ref`.
- **verdict_scope:** FORMULATION (§3.5 plateau across 4 solvers) · INSTANCE (the burn is one realization,
  n600, not yet eval'd). §3.1–§3.4 BUILT + unit-tested + n8/n16-smoke; the n600 measurement IS the burn.
- **next:** MAIN's post-burn chain (endpoint EMA ckpt → pose re-solve tt1 twin → photometric → TTO → gate
  → exact eval). Falsifier (sg1 Contrarian): re-burn endpoint d_seg ≥ cell_drop50's 0.004310 at matched
  bytes → coarse-from-birth closes at INSTANCE, solve-distillation (QA75) leads.
