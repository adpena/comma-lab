---
title: DDM CC2 quantizer and per-counted-stream coder races DAG FEED
utc: 2026-07-25T03:59:00Z
lane_id: ddm_cc2_coder_races
verdict: ADVISORY_RACES_COMPLETE_RECEIVER_INTEGRATION_OWED
research_only: true
score_claim: false
pointer_moved: false
main_landing_review_required: true
---

# Executable DAG

`W_joint source 138801 B @ 5aa45850...`
→ `J8F Step-4 theta @ f6e88a74...`
→ three deterministic terminal quantizer operators
→ real W_joint compile/parse-back
→ exact active zero-effect `PC1 40 B @ 05a98fac...`
→ `139538 B` counted compositions
→ Q8 verdict reuse only by exact SHA equality
→ fresh C3/v5 `camera -> uint8/R -> frozen scorers` n600
→ `v5 proxy delta action = -0.001924772136713493`
→ **advisory Race 2 winner; no promotion or pointer movement**.

In parallel:

`exact Q8+PC1 composition 139538 B`
→ recursive stored-ZIP physical-leaf expansion
→ `27 leaves + 5292 B fixed container overhead`
→ five exact same-object coder arms per leaf
→ explicit model/header/stream accounting
→ canonical parse-back for all 135 frames
→ typed c1 waterfill/costate SENSE
→ `136116 B` mixed estimate (`-3422 B`)
→ **receiver integration owed; no archive or score claim**.

# FEED

- Typed config: `.omx/research/configs/ddm_cc2_coder_races_20260725.json`.
- Reusable coders: `src/tac/optimization/arith_selfcomp_rate_coders.py`.
- Reused MS7 race surface: `src/tac/optimization/ddm_ms7_receiver_edges.py`.
- Typed race/pricing core: `src/tac/optimization/ddm_cc2_coder_races.py`.
- Resumable runner: `tools/run_ddm_cc2_coder_races.py`.
- Repository receipt: `.omx/research/ddm_cc2_coder_races_receipt_20260725T035900Z.json`.
- Findings: `.omx/research/codex_findings_ddm_cc2_coder_races_20260725T035900Z_codex.md`.
- Full external receipt: `/Volumes/VertigoDataTier/pact/experiments/results/ddm_cc2_coder_races_20260725T030606Z/ddm_cc2_coder_races_receipt.json` (`f9432959...`).
- Downstream: MAIN review decides whether to integrate only the eight negative-delta leaf coders and whether a true multi-seed/v5 schedule trial is authorized.

# Triality

- DSL: strict JSON config binds source/checkpoint/PC1/cache/source-harvest bytes and hashes, SSD output, seed, and batch size.
- DAG: atomic stage checkpoints `00` through `05`, exact-SHA reference reuse, fresh scorer arms, then lossless per-leaf pricing.
- Equations: `S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489`; lossless Race 3 rows have `delta d_seg = delta d_pose = 0`, so `delta S = 25*delta bytes/37545489`.

# Pointer delta honesty

The work is `[macOS-CPU advisory]`, `score_claim=false`, `promotion_eligible=false`. Pointer `0.1910828242 [contest-CPU]` did not move. The Race 3 integrated archive remains owed, so the derived byte estimate cannot be presented as a candidate.

# STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; craft handoff; v7.5 contract; canonical pointer surfaces; live inbox; J8F, PC1, MS7 receipts and exact bytes; C3/Cool-Chic SHA-bound source harvest; n600 target cache; frozen upstream scorer.
