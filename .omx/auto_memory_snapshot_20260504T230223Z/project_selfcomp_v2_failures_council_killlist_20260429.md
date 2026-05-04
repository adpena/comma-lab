---
name: Selfcomp paradigm v2 failures + council kill list (2026-04-29 PM)
description: All 4 Selfcomp lanes (MM, SA v2, SC++ v2, SO v2) failed first-dispatch on Modal T4. Dead-flag scanner false-positive bug (MM rc=3 in 4s) + CUDA OOM (SA/SC++/SO rc=1, 7GB needed on 14GB T4 with 11GB used). Fixes landed; v3 dispatched on A10G; SO killed per council. 5 sweep + 5 EUREKA lane scripts ready, deferred.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Failures and root causes**:

- **Lane MM** (rc=3 in 4s): inline Python -c dead-flag scanner regex matched the line-12 comment mention of `experiments/build_lane_mm_archive.py`, then captured `--hard` from line-23 comment "NEVER git pull / git reset --hard." Every dispatch failed at scan stage. FIXED commit 614cd9f3 (removed inline scan).
- **Lane SA v2** (rc=1 in 126s): SegMapTrainer.train_epoch processes ALL 600 pairs in one forward; tries to allocate 7.03 GiB on T4 (14.56 total, 11.66 already used by GT/scorers, 2.90 free). `train_segmap.py:355` calls `train_epoch(mask_pairs=mask_pairs)` passing the full tensor; --batch-size CLI flag is parsed but unused inside the trainer.
- **Lane SC++ v2 / SO v2**: same OOM root cause as SA.

**Permanent fixes in progress (Subagent H)**:
- Extend `tac.preflight.preflight_arity` to walk shell-script `("$PYBIN"|python) -u? experiments/<file>.py` invocations and validate flags. Closes the bug class shell-side.
- Add proper batch chunking to `SegMapTrainer.train_epoch` with `batch_size` kwarg. Wire `train_segmap.py --batch-size` through.
- Add STRICT preflight check that flags T4 dispatches with potentially-OOM training scripts.

**Re-dispatched on A10G (24GB)**:
- Lane MM v2: `fc-01KQD283JJFDGJ11XYPPQK7VN5` (T4, encoder-only)
- Lane SA v3: `fc-01KQD2AHSSJEENJB2WBAKB2XRA` (A10G)
- Lane SC++ v3: `fc-01KQD2AKKV97GVWXYYQVKPM84W` (A10G)
- Lane SO v3: `fc-01KQD2AN7CKKHW9H2JEYMZQRKF` — **CANCELLED per council kill list**

**Council kill list (codex grand council 2026-04-29 11am, gpt-5.5 xhigh)**:
1. **Lane SO**: Hessian-aware code path falls back to default block-FP (script doesn't actually spend bits per Hessian — confirmed in `pack_payload_tar_xz(state, '$PAYLOAD')` fallback line). Self-deception. CANCEL until exporter is real.
2. **Lane SA v3 → kill once SC++ v3 produces viable mid-train checkpoint** — SA is dominated by SC++ (KL distill = expected -0.05).
3. **Defer HM-S, WC-S, MAE-V, SAUG** until SC++ control [contest-CUDA] lands.
4. **Reframe Lane FR-Ω as EXPORT VARIANT on SC++ best checkpoint, not new 12h training** — saves ~$4/10h.
5. **Restrict Lane DARTS-S to 3 configs** (default 24×24×8, wide 32×32×8, deep 24×24×12) — only after SC++ control.

**How to apply**:
- Watch SC++ v3 (A10G ~12-14h) — when first checkpoint lands, kill SA v3 and start FR-Ω/SH/PD as cheap export variants.
- DON'T dispatch HM-S/WC-S/MAE-V/SAUG until SC++ score lands.
- Watch the OOM-fix subagent (#H); once batch-chunking is live, can re-dispatch SC++ on T4 as cost optimization.

Cross-refs:
- project_grand_council_brutal_forecast_20260429
- project_selfcomp_portfolio_tonight_20260429
- project_selfcomp_reverse_engineered_20260429
