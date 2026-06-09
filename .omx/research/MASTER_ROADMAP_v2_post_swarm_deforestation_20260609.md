# MASTER ROADMAP v2 — post-swarm + deforestation layer (all vehicles in parallel)

UTC: 2026-06-09 · claude · supersedes-by-extension `MASTER_ROADMAP_hinerv_snerv_interpreter_compiler_20260609.md`
(v1 PRESERVED per append-only). Frontier (pointer, never hardcoded):
**0.19199 [contest-CPU]** / **0.20533 [contest-CUDA]**. S = 100·d_seg + √(10·d_pose) + 25·bytes/37_545_489.

## What changed vs v1
- **Phase A CLOSED by the 4-agent swarm** (disjoint ownership, serializer, NO-FAKE tests):
  - A1 tie-corrector torch-EXACT authority in birth/survival — `bce2bda0c` (V1-TIE).
  - A2 `CandidateActionEvaluation` as admission law: export/pack `c350effab` (V1-EXPORT) +
    lowering-race/launch-gate/producers `7cae07c0c` (V3-COMPILER).
  - A3 3 stale gate-vocabulary tests fixed (each a distinct real root cause) — `bce2bda0c`.
  - C1 SNeRV TUB DROP_OR_REIFY proof — `06f3dc580` (V2-SNERV): output_2 DROP (receiver RGB
    bit-identical ⇒ eliding rent-optimal), yl_norm REIFY_PENDING_SCORER.
- **NEW strategic layer: DEFORESTATION** (operator 2026-06-09: "give everything what it needs
  and no more; tree-shake/slash-and-burn to the skeleton of only what's necessary in the
  optimal format and quantization"). The scorer read-surface (code-grounded, drift-checked in
  `contest_eval_contract`) is the burn map:
  - SegNet reads ONLY `x[:,-1]` (2nd frame of each pair) ⇒ ~600/1200 frames are SEG-FREE
    (only need pose fidelity).
  - SegNet is argmax-only ⇒ per-pixel precision = top-2 logit margin (boundary protected,
    interior free) = per-pixel waterfilling.
  - PoseNet scores only 6 of 12 dims ⇒ half the head is null space.
  - Vehicle 3 compresses the EVALUATOR-EQUIVALENCE CLASS, not RGB; inflate.sh emits an
    evaluator-inverse render (engineered to hit cells, not look real). Compliance boundary:
    must remain real RGB frames the scorer runs on.

## The three parallel vehicles (current launch)
- **V1 HiNeRV (Phase B) — THE FRONTIER SHOT.** B0 timing smoke (free local MLX, MVP-first) →
  B1 600-pair PR95 curriculum (L14-L17: 8-stage 29650-ep, Muon final, C1a, sigma, EMA 0.997,
  eval_roundtrip, score-aware loss) → B2 dual CPU+CUDA exact eval → B3 verdict vs 0.19199.
  B0 is the gating measurement (turns unknown GPU cost into measured hours).
- **V2 SNeRV (Phase C) — parallel lane.** C1 done. C2 MFU/HFR/TUB source-forward binding +
  C3 LF/HF byte-pressure → C4 exact eval. Each receiver-causal + uint8/scorer-surviving facet
  emits a base-bound CandidateActionEvaluation (rent currency) to enter LoweringRace.
- **V3 Compiler/deforestation (Phase D-primitives).** Read-surface-grounded atom proposers:
  (1) per-pixel argmax-margin tolerance map; (2) even-frame-seg-free classifier; (3) pose-null
  projection. B-before-D: PRIMITIVES now (feed B1 score-aware loss + the post-B2 waterfiller),
  NOT the live loop (waits for B2 base per Hotz council SEAL).

## Disjoint ownership (this launch)
- V1-B0: CREATE `tools/timing_smoke_hinerv_pr95_family.py` (+ output JSON/memo). READ-ONLY all else.
- V2-SNERV-C2: `snerv_official_source_forward_harness.py`,
  `snerv_official_tub_lf_hf_replacement_authority_gate.py`, `snerv_lf_hf_replacement_queue.py` + tests.
- V3-DEFOREST: CREATE `src/tac/optimization/scorer_read_surface_atoms.py` + test. READ-ONLY all else.

## Cross-cutting (unchanged from v1)
Rent law (admit iff S(base+σ)<S(base)); anti-drift base-binding; dual CPU(Linux x86_64)+CUDA(T4)
for authority (never MPS/macOS-CPU); tie-corrector on every authority decision; serializer +
--expected-content-sha256 + review-gate (2 clean .py passes); SSD disk hygiene, no /tmp evidence;
NO FAKE. Submit only if exact dual-axis S beats the public frontier.
