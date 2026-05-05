---
name: Lane 12 (NeRV mask codec) READY-TO-DISPATCH plan + Phase F empirical = 94.4% byte savings on REAL Lane G v3 masks
description: 2026-04-30. Lane 12 promoted from Level 1 (synthetic scaffold) → Level 2 (real-archive empirical). Phase F empirical: 23,594 B NeRV-fp16 vs 421,483 B AV1-baseline = 94.4% byte savings AND 2.0% argmax disagreement after only 1400 CPU training steps. Phase G (contest-CUDA dispatch via scripts/remote_lane_nerv.sh) is READY-TO-LAUNCH on Vast.ai 4090 — operator click required.
type: project
originSessionId: lane-12-promotion-subagent
---

## TL;DR

**Lane 12 (NeRV mask codec) is at Level 2 INTEGRATION**: end-to-end codec + trainer + inflate dispatch + remote_lane script + Check 95 STRICT preflight + 34 unit tests + 9 integration tests, all green. Real-archive empirical on Lane G v3 masks shows **94.4% byte savings** (23,594 B NeRV-fp16 vs 421,483 B AV1) with 2.0% argmax disagreement after only 1400 CPU steps.

Phase G (contest-CUDA dispatch) is **READY-TO-LAUNCH** but requires operator action (Vast.ai launcher), so this memory tags Lane 12 as Level 2 → Level 3 PENDING dispatch.

## Empirical numbers (Phase F, [empirical:reports/lane_12_nerv_real_archive.json])

| Metric | Value | Comment |
|---|---|---|
| AV1 baseline (Lane G v3 masks.mkv) | **421,483 B** | extracted from `experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip` |
| NeRV-fp16 (NRV2 payload) | **23,594 B** | hidden=64 depth=4 num_freqs=8 (11,781 params); council Phase B verdict |
| NeRV-int8 (NRV2 payload + scale table) | **11,845 B** | predicted Phase A2 stretch (currently scaffold-tested only) |
| Byte savings vs AV1 (fp16) | **94.4%** | 17.9× reduction |
| Argmax disagreement vs AV1 source | **2.0%** | partial training (1400 of 60000 steps); CUDA full run predicted ≤1% |
| Training steps (CPU partial) | 1400 | hit 200s wall clock; full 60000 needs CUDA |
| Predicted final disagreement after full CUDA training | **<1%** | scaling from CPU loss curve: 0.59 → 0.02 in 1400 steps continuing → ≤0.005 expected |

## What's at Level 2 (committed)

| File | What it does | Test coverage |
|---|---|---|
| `src/tac/nerv_mask_codec.py` | NeRVMaskCodec + NeRVMaskTrainer + NRV2 wire format (fp16 + int8-with-scale) + render_mask_argmax | 24 tests in `test_nerv_mask_codec.py` |
| `experiments/train_nerv_mask.py` | Standalone trainer: SegNet-extract → train → encode → write provenance | smoke-tested CPU run |
| `submissions/robust_current/inflate_renderer.py` | `_load_masks_from_nrv` + dispatch via `.nrv` extension AND `b"NRV1"` magic-byte sniff | works with synthetic + real-archive payload |
| `scripts/remote_lane_nerv.sh` | Canonical 4-stage Vast.ai dispatch: NVDEC probe → train → archive build → CUDA auth eval | shell-syntax-clean, all 14 STRICT remote_lane checks pass |
| `src/tac/profiles.py` | `NERV_MASK_LANE_G_V3` profile entry | registered in `PROFILES["nerv_mask_lane_g_v3"]` |
| `src/tac/preflight.py` | `check_nerv_codec_uses_ema_and_no_mps_and_auth_eval` (Check 95) STRICT @ 0 violations | 10 tests in `test_preflight_nerv_codec_discipline.py` |
| `.omx/research/lane_12_nerv_scaffold_audit_20260430.md` | Phase A audit | — |
| `.omx/research/council_lane_12_nerv_design_20260430.md` | Phase B council deliberation (UNANIMOUS GREEN) | — |
| `reports/lane_12_nerv_real_archive.json` | Phase F empirical measurement | tagged `[empirical:reports/lane_12_nerv_real_archive.json]` |

## CLAUDE.md non-negotiables verified

- ✅ EMA: trainer instantiates `tac.training.EMA` (canonical class) at decay 0.997. `update()` after every `optimizer.step()`. `encode()` ships the EMA shadow, NOT live weights.
- ✅ eval_roundtrip: trainer uses cross-entropy on raw 5-class logits (NO `.round()` zero-gradient bug). The 384→874→uint8→384 SegNet eval roundtrip is delegated to `experiments/contest_auth_eval.py` Stage 3.
- ✅ MPS refused at trainer + render_mask_argmax construction (matches SegMapTrainer pattern).
- ✅ Auth eval at end of chain: `scripts/remote_lane_nerv.sh` Stage 3 invokes `experiments/contest_auth_eval.py` against the rebuilt archive.
- ✅ No bare `.round()` in any forward / step / sample / evaluate method (Council A bug class). Self-test enforces this in source.
- ✅ No silent defaults: every public function arg required-keyword or explicit; trainer construction raises on `None` codec, ema_decay outside (0,1).
- ✅ Strict scorer rule: `inflate_renderer.py:_load_masks_from_nrv` loads only the small NeRV codec at inflate; NO SegNet/PoseNet.
- ✅ Commit serializer: all commits via `python tools/subagent_commit_serializer.py`.
- ✅ Subagent-friendly: no bare `Bash run_in_background:true` for >3min jobs (training was foreground bash with timeout=300s).

## Phase G dispatch plan (READY-TO-LAUNCH; operator action required)

### Cost estimate

- Vast.ai RTX 4090 @ $0.25/hr × ~3h training + 0.5h auth eval = **~$0.85 total** (well under $10 unattended cap; still well under $25 Vast.ai budget cap).
- Modal A10G/T4 alternative: ~$1.20-1.50 (T4 is ~5× slower; A10G has OOM risk per CLAUDE.md if memory budget mis-set — NeRV is tiny so OOM-safe but T4 is fine).

### Recommended platform: Vast.ai RTX 4090

```bash
# Operator runs from local machine (not subagent):
python tools/launch_lane_on_vastai.py \
    --lane lane_12_nerv \
    --script scripts/remote_lane_nerv.sh \
    --label lane_12_nerv_$(date +%Y%m%dT%H%M%SZ) \
    --gpu RTX_4090 \
    --max-cost 5.0 \
    --filter 'gpu_name=RTX_4090 reliability>0.95 inet_down>200 disk_space>30 num_gpus=1'
```

The script writes `lane_12_nerv_results/RESULT_JSON` containing the contest-CUDA score on the rebuilt archive (Lane G v3 anchor with masks.mkv replaced by masks.nrv).

### Predicted band [prediction]

- bytes (deterministic): **23,594 B** (matches Phase F local) → archive total ~336 KB
- score: **[0.95, 1.30]** [contest-CUDA]
  - Lane G v3 anchor: 1.05 [contest-CUDA]
  - NeRV mask-only swap: 0.04 saved on rate term (424 KB → 24 KB → 25 × 0.000400/37545489 = -0.00000027 ... wait, recompute) — actually 25 × (337,000 - 421,000 + 24,000) / 37545489 ≈ 25 × -60K / 37.5M ≈ -0.04 score on rate alone
  - SegNet score from 2% mask disagreement: predict +0.01 to +0.04 (boundary pixels are the dominant loss term — UNIWARD-style boundary weighting is Phase A2)
  - Net: predicted in [0.95, 1.05] if disagreement converges to ≤1% under full training; widens to [1.05, 1.30] if boundary underfit catastrophic
- ESTIMATE WIDTH justified: this is the FIRST contest-CUDA NeRV-mask-codec measurement — no prior data point.

### Kill criteria (binding per Phase B council)

1. NRV bytes > 100 KB → abandon (worse than 50% AV1 savings would defeat the lane)
2. SegNet score > Lane G v3 + 25% (1.05 × 1.25 = 1.31) → abandon (boundary underfitting catastrophic)
3. Inflate render time > 30s on T4 → abandon (would exceed 30-min total inflate budget when stacked)

### Post-dispatch operator actions

1. Harvest within 24h: `tools/harvest_modal_calls.py` if Modal; auto-survives on Vast.ai.
2. Tag the result: `[contest-CUDA] reports/lane_12_nerv_cuda.json` (or `[KILLED]` per kill criteria above).
3. If GREEN, update this memory file with the actual score + bytes; promote Lane 12 from Level 2 → Level 3.
4. If RED, document the failure + which kill criterion fired; codec stays Level 2.

## 3-clean-pass adversarial review counter

- Round 1 (implementer self-review): CLEAN. All CLAUDE.md non-negotiables verified, 34 tests pass, real-archive empirical, dispatch script syntax-clean. Counter → 1/3.
- Round 2 (Quantizr / van den Oord / Hotz adversarial): CLEAN. All theoretical objections explicitly addressed by Phase A2 stacking flags + kill criteria + empirical evidence. Counter → 2/3.
- Round 3 (Shannon / Dykstra / Yousfi math-rigor): **ISSUE FOUND** — Dykstra D2: council verdict needed strategic-framing clarification ("stacking lane" not "standalone winner"). FIX LANDED in council Phase B doc. Counter → **0/3 RESET**.

**Status**: 3 NEW clean passes required by the next session that picks up Lane 12 post-Phase-G result. This subagent's session leaves Lane 12 at:
- **Level 2 INTEGRATION** (committed) — codec + trainer + inflate dispatch + remote_lane script + Check 95 STRICT preflight + 34 tests + real-archive empirical
- **3-clean-pass counter at 0/3** (Round 1+2 clean, Round 3 found a doc issue and reset)
- **Phase G CUDA dispatch READY-TO-LAUNCH** (operator action required, ~$0.85 estimate)

## Cross-refs

- Audit: `.omx/research/lane_12_nerv_scaffold_audit_20260430.md`
- Council design: `.omx/research/council_lane_12_nerv_design_20260430.md`
- Phase F empirical: `reports/lane_12_nerv_real_archive.json`
- Maturity standard: `feedback_production_hardened_standard_definition_20260430.md`
- Phase 2 Lane 12 design: `project_phases_2_3_4_design_implementation_math_provenance_20260429.md` § "Lane 12"
- Production-hardened canonical paths: CLAUDE.md "EMA", "eval_roundtrip", "Auth eval EVERYWHERE", "MPS auth eval is NOISE", "Strict scorer rule"
