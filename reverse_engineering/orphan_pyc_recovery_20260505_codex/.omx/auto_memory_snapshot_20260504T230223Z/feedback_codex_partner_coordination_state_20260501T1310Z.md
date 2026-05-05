---
name: Codex partner coordination state — 2026-05-01T13:10Z (Lightning T4 queue + Alpha matrix verdict + 4 Claude lanes acknowledged)
description: Live coordination snapshot from codex's research ledger update at 13:10Z. Codex confirms 4 Claude orthogonal lanes in flight on Vast 35959478, runs Lightning T4 promotion queue, lands Alpha matrix verdict (pure-symbol RLE > 902KB > current 421KB masks.mkv → not the breakthrough path), and has component-response stack optimizer + custom decoder scaffold landed. Cross-handoff rules: codex Alpha primitive evidence → claude sparse encoder; claude ADMM/stack archives → codex Lightning T4 promotion candidates after local custody review.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## Codex-owned scopes (do not duplicate)

- Lightning T4 promotion + harvest for exact archives
- Alpha primitive response plans + CUDA component-response measurement
- Component-response stack optimization + Dykstra-style planning (`experiments/optimize_component_response_stack.py`)
- Custom decoder + overfit codec planning (Rust/Zig/C/static binary, Python reference, lossy geometry + charged sparse repair, temporal grammar, INR/NeRV handoff, RL/bandit payload search)
- Durable DX fixes (Lightning job identity/path normalization, supply-chain guards, deterministic staging, MCP process cleanup)

## Claude-owned scopes (acknowledged by codex)

- Orthogonal owv3_0120 stack composer on Vast 35959478 (subagent a6353c42db64459de)
- NeRV/HNeRV/SNeRV mask codec real training (subagent a9fbed6779dd8ef6d)
- Joint-ADMM cross-stream coordinator on owv3_0120 (subagent af4c4b239bf169a05)
- PoseNet-aware sparse overfit encoding using scorer-Jacobian sensitivity (subagent a00c0ed04712afe20)

## Lightning T4 queue snapshot (13:10Z)

| Job | State | Notes |
|---|---|---|
| `exact_eval_alpha_crf63_grayscale_t4_20260501T125258Z` | Running | Alpha grayscale baseline, fastest large-rate signal |
| `exact_eval_owv3_0119_wave3_t4_20260501T125143Z` | Running | T4 promotion of claude's owv3_0119 (1.0027 RTX 4090 already landed) |
| `exact_eval_owv3_0120_wave3_t4_*` | Pending | T4 promotion of claude's owv3_0120 (1.0024 RTX 4090) |
| `component_response_alpha_primitive_pfp16_crf63_t4_20260501T130822Z` | Pending | Alpha primitive component-response measurement |

Note: 0065/0032 Wave3 jobs were NOT durable in `.omx/state/lightning_batch_jobs.json` despite earlier terminal output. Codex flagged them as needing explicit resubmission.

## Alpha matrix verdict (NEW evidence)

Per `experiments/results/alpha_mask_codec_candidate_matrix_pfp16_20260501_full/`:

| Encoding | Bytes vs current masks.mkv (412,169 compressed / 421,483 raw) |
|---|---|
| COCO-style foreground RLE | 902,161 bytes (2.2× regression) |
| Transition endpoints | 997,410 bytes (2.4× regression) |
| Component boundary delta | 1,476,543 bytes (3.6× regression) |

**Interpretation**: exact naive RLE/transition/component packets are NOT the breakthrough path. The viable path is **lossy geometry + charged sparse repair** OR **neural INR/NeRV** OR **entropy-coded temporal grammar**, all gated by CUDA component-response measurement.

This validates Claude's NeRV mask codec subagent (a9fbed6779dd8ef6d) — neural INR is exactly the lossy-geometry path codex's matrix verdict points toward.

## Codex landed implementation

- `experiments/custom_mask_codec_probe.py` — `CMCP_RLE1` probe scaffold (empirical only, non-promotable per codex)
- `src/tac/tests/test_custom_mask_codec_probe.py`
- `docs/runbooks/custom_decoder_overfit_codec_plan_20260501.md` — overfit codec plan
- `experiments/optimize_component_response_stack.py` — Dykstra-style optimizer (rejects non-CUDA/noncanonical/prediction-only inputs for promotable mode)
- `src/tac/tests/test_optimize_component_response_stack.py`
- Lightning SDK fix: `lightning_sdk_job_name()` now lowercases after underscore→hyphen normalization (matches `/teamspace/jobs/exact-eval-...t130313z/artifacts`)

Verification: 15 focused tests passed (Alpha matrix, custom mask codec probe, stack optimizer, Lightning SDK normalization). MCP cleanup clean.

## Cross-handoff rules

1. Codex Alpha primitive response evidence → Claude's sparse encoder subagent (a00c0ed04712afe20). The per-pixel Jacobian Claude is computing IS the empirical perturbation gradient codex's plans estimate; convergent design.
2. Claude's ADMM/stack archives (subagents 1+3) → codex Lightning T4 promotion ONLY after local custody review (archive SHA, payload closure, component distances, sample count, CUDA device, canonical path).
3. Claude's NeRV mask codec (subagent 2) → competes with codex's lossy-geometry/charged-sparse-repair direction; both feed component-response optimizer.

## Score state at this checkpoint

- Deploy champion: **owv3_0120 (1.0024 [contest-CUDA RTX 4090])** locked at commit 93be3a1c
- T4 promotion of 0120 still pending (queued by codex) — A++ grade requires T4-equivalent + adversarial review
- Wave-4 cliff confirmed at bbr ~0.60; all 5 deep-bbr candidates regressed (commit 61880403)

## Build Discipline directive (just landed at 377cf144)

User directive: "all should be built with a mind for OSS and eventual paper and production deployment at comma-ai and stacking and composability and everything". Codified in AGENTS.md "Build Discipline — OSS, Paper, Production, Composability" section. The four constituencies:
1. OSS-readiness (clear modules, type hints, docs, no embedded creds)
2. Paper-readiness (every claim has [evidence:<artifact>] tag + ledger entry)
3. comma-ai production-readiness (T4 + 30 min budget, deterministic decoders, reproducible Rust/Zig builds)
4. Stacking and composability (canonical order rep→pred→quant→hyper→arith→pack, typed contracts)

Subagents already in flight will need post-landing OSS/paper/composability audit since they were dispatched before this section landed.

## Cross-refs

- `project_shannon_floor_codex_partner_coordination_20260501.md` (the canonical coordination protocol)
- `.omx/research/shannon_floor_nextwave_telemetry_and_research_20260430_codex.md` (codex live progress ledger; +69 lines added at 13:10Z)
- `project_lane_g_v3_owv3_0120_LANDED_1_002_20260501.md` (deploy champion)
- `AGENTS.md` "Build Discipline" section (commit 377cf144)
