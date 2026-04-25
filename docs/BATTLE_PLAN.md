# Battle Plan: Beat Quantizr (0.33)

**Deadline: May 3, 2026. 8 days remaining.**
**Last verified contest-compliant auth: 2.01 (April 21, full e2e through inflate.sh → upstream evaluate.py).**
**Target: < 0.33. Realistic stretch this cycle: 1.0–1.5 with current architectures.**

> **IMPORTANT — score honesty:** the previously cited "auth 2.26" came from a SHIRAZ proxy of 0.804 translated through a chain that contained the Round 22 QAT arch-mismatch bug. It is NOT a verified contest-compliant score. The last fully measured number is 2.01 (April 21, post mask-resolution fix).

## TODAY (Apr 25) — NUCLEAR (zero GPU)

These actions move the score the most per dollar. Do them BEFORE any new training.

> **R27 correction:** prior battleplan double-counted NUCLEAR #1 and #3 — they are TWO SIDES of one architectural change (store odd frames + reconstruct even at inflate). Combined delta is −0.12 to −0.18 once, not twice.

| # | Action | Cost | Expected delta | Blocker |
|---|--------|------|----------------|---------|
| **1+3 (combined)** | **Half-frame mask + even-from-odd reconstruction**: store only 600 odd-frame masks in archive; at inflate, reconstruct even-frame masks from stored ones via grid_sample warp using stored pose. Quantizr's paradigm. Existing `inflate_renderer.py` half-frame path uses `repeat_interleave` (duplication) which zeroes the MotionPredictor's diff features — the warp version preserves motion. | CPU, ~1 day (encode sweep + warp impl + e2e validation) | rate −0.12 to −0.18 (one delta, not two) | None — int8 overflow already fixed in `encode_masks_monochrome` (R27 reviewer correction) |
| **2** | **SHIRAZ Phase 3 → immediate auth eval** (download checkpoint, run inflate → upstream evaluate.py). Do NOT wait for QAT chain to finish. Get the auth number BEFORE deciding next deploy. | local, ~3h | establishes ground truth | SHIRAZ Phase 3 complete |
| **4 (NEW NUCLEAR — promoted from "open gaps")** | **Add `--profile X` to `train_distill.py` + deterministic seeding (cudnn, numpy, random) + add CI to run preflight tests on every push.** These are CLAUDE.md non-negotiables and NOT council decisions. Without them, every experiment is unreproducible by definition. | local, **4–6h** (timebox strictly; CI alone is 1–2h on a fresh repo + iteration) | unblocks 5-pass review gate | None |

Combined projection of NUCLEAR #1+3 + #2 baseline: rate term drops by 0.12-0.18 from 9.4-point contribution.

## Round 23 + 24 Postmortem (2026-04-25)

The SHIRAZ A100 trained with `motion_hidden=24, depth=1`, but `pipeline.py` invoked `qat_finetune.py` without `--motion-hidden`/`--depth`/`--embed-dim`. QAT silently rebuilt the wrong arch. Two rounds of fresh-eyes adversarial review surfaced 14+ findings; all CRITICAL ones are fixed:

**Round 23 fixes (10):**

| # | Bug class | Fix |
|---|-----------|-----|
| 1 | Cross-function `cmd` variable scope pollution in arity validator | `_extract_invocations_from_scope` per-scope `list_vars` |
| 2 | Short-form alias (`-m, --motion-hidden`) not indexed | `_parse_argparse_signature` iterates all positionals |
| 3 | `ARCH_FLAGS_BOOLEAN` declared but never checked → boolean-flag SHIRAZ | Rule D in `preflight_arity` |
| 4 | `bash -c "python experiments/x.py"` was a blind spot | `_BASH_C_TARGET_RE` |
| 5 | `optimize_poses.py` `.pt` loader missing `padding_mode` + `use_dilation` | Pass through from `model_cfg` |
| 6 | `step_qat` skip path returned hardcoded filename → phantom path | Disk-resolve actual artifact |
| 7 | Stale `run_pipeline.sh` survives rsync on remote | `deploy_vastai.py` preflight check 6 |
| 8 | New tests at `src/tac/tests/` not in `pyproject.toml testpaths` | `testpaths = ["tests", "src/tac/tests"]` |
| 9 | `preflight_profiles` silently skips profiles with no `experiment_type` | Fail on missing/unknown |
| 10 | Validator error referenced deleted `preflight_codebase.py` | Message updated |

**Round 24 fixes (5):**

| # | Bug class | Fix |
|---|-----------|-----|
| 11 | `step_export` built `AsymmetricPairGenerator` without `use_zoom_flow` → silent wrong arch on GREEN | Pass `use_zoom_flow`; load_state_dict mismatch now hard-errors |
| 12 | `PipelineConfig` missing `use_zoom_flow` (and other discipline flags) → `getattr` fallbacks masked omission | Real fields, removed `getattr` |
| 13 | SSH check 6 returned empty on SSH failure → false-negative | Sentinel-token compound check |
| 14 | `step_eval` cached `.done_eval` on auth-eval failure → phantom score on resume | Raise instead of mark done |
| 15 | I4LZ format has no arch header → contest-time silent default `padding_mode=zeros` for non-default arches | `step_compress_weights` falls back to FP4 when arch needs header |
| 16 | `ARCH_FLAGS_BOOLEAN` set incomplete (missing `--use-swa`, `--beneficial-quant-noise`, freeze flags, etc.) | Set expanded; Rule D now flags 5 real launcher gaps in `step_fridrich_refine` (fixed) |

**Tests pass: 46/46** (`src/tac/tests/test_preflight_arity.py` + `test_integration_boundaries.py`). R28 added 2 more tests for the score regex distortion-keyword check + _user_provided_flags required-arg semantic, and strengthened the profile-matrix check to round-trip through PipelineConfig (catches type mismatches).

## Open architectural gaps

R27 reviewer correction: items 1, 2, 4 are CLAUDE.md non-negotiables and have been promoted to NUCLEAR #4 above (timeboxed, ~4h total). Items 3, 5, 6 remain open:

3. **Profile-key vs ArchConfig-field cross-validation.** R26 added Levenshtein typo detection, but a profile key that's training-only AND happens to be far from any PipelineConfig name is silently dropped. Acceptable for now (training scripts consume their own profile keys), but `preflight_profiles` should grow a similar warning for `ArchConfig.__dataclass_fields__`.
5. **Chain bugs in pipeline.py:** sensitivity_sweep runs on `cfg.checkpoint` which mutates across iterations (silent semantic drift); fridrich_refine runs AFTER QAT (should be before — Fridrich sculpts float weights into scorer null-space, then QAT bakes the sculpting in); `subprocess.run` has no timeout (no preemption recovery).
6. **No `download` subcommand on deploy_vastai.py.** After pipeline completes, results in `/workspace/pact/experiments/results/<profile>/` require manual SCP. Risk: operator destroys instance before downloading. Add `python scripts/deploy_vastai.py --download <instance_id>` and gate `--kill` behind a confirmation prompt.

## Re-opened HOLD items (R27 reviewer correction)

The earlier KILL list was too aggressive. Two items are restored to HOLD pending fix:

- **Cool-Chic / C3 residual**: KILL was based on "FP4 quantization erases float gains." But the run log shows C3 float-path SegNet improving from 92.3 → 68.7 over 20 epochs — real signal. The FP4 export is broken (QAT parametrization buffers not on training device), not the architecture. Status: HOLD until FP4 export bug is fixed. Then re-evaluate.

## Experiment ranking (Round 24 Contrarian decision)

| Tier | Experiment | Justification |
|------|-----------|---------------|
| **NUCLEAR** | Half-frame mask AV1 sweep | CPU-only; largest single rate reduction available |
| **NUCLEAR** | SHIRAZ auth eval immediately on Phase 3 exit | Truth before next deploy; do NOT skip to keep momentum |
| **NUCLEAR** | Even-frame-warp-from-odd at inflate | Zero archive cost; multiplicative with mask sweep |
| **PROMOTE** | MXLZ sensitivity sweep + mixed-precision export on winner | rate −0.06 to −0.09; requires arch_header guard (just added) |
| **PROMOTE** | Engineered corrections | SegNet −0.05 to −0.10; only after auth baseline known |
| **PARALLEL** | `winner_v2` (SHIRAZ_V2 or WILDE_V2) | beneficial_quant_noise has ZERO empirical backing; only deploy after winner auth is known |
| **HOLD** | GREEN auth eval | A/B for zoom_flow ablation; passive value |
| **KILL** | Cool-Chic full, C3 full | FP4 quantization erases float gains; not solved |
| **KILL** | DP-SIMS revival, all VQ-VAE, diffusion teacher, distillation full | No auth evidence; no time |
| **KILL** | All `cross_disc_*`, `finance_*`, `variational_*`, `lagrangian_dual_*` | Theory-only; not submission candidates |
| **KILL** | `OVERFIT_CPU/GPU`, `wavelet`, `coord`, `depthwise`, `channel_recurrent` smokes | No auth evidence; no time |

## Rebuilt Timeline

| Day | Action |
|-----|--------|
| Apr 25 (today, AM) | Half-frame AV1 sweep (CPU, parallel with everything else). Even-from-odd inflate code. |
| Apr 25 (today, PM) | SHIRAZ Phase 3 auth eval. Diagnose proxy-auth gap if > 1.5x. Decide v2 deploy based on result. |
| Apr 26 | Deploy winner_v2 if SHIRAZ auth is in [0.5, 1.2]. MXLZ sensitivity sweep on winner. Engineered corrections on winner. |
| Apr 27 | v2 results. Stack: half-mask + even-from-odd + MXLZ + engineered. Full e2e auth eval. |
| Apr 28-30 | **5-pass adversarial review starts NOW (not May 1).** Review inflate.sh, archive packaging, contest compliance, T4 timing. Each round resets on any finding. |
| May 1-2 | Final polish + buffer. |
| May 3 | Deadline. |

> **Contingency** (R27 reviewer): Rounds 23-27 each found CRITICAL bugs (zero clean rounds so far). 5 consecutive clean passes by May 1 is statistically implausible at the current find-rate. **If the gate is not clean by May 1, the submission is whatever passes review by May 3 OR is withheld.** A non-compliant submission ranks worse than no submission per the writeup strategy. Quality of the gate is more important than meeting the original timeline.

## Score Projections (honest, R27 math correction)

| Scenario | Score | Path | Confidence |
|----------|-------|------|------------|
| Verified baseline (Apr 21) | 2.01 [contest-compliant] | masks 384×512 + ASYM renderer + poses | HIGH |
| SHIRAZ Phase 3 auth (pending) | unknown | A100 7-stage chain | proxy says 0.8, auth gap unknown |
| + half-frame + warp (NUCLEAR #1+3, ONE change) | −0.12 to −0.18 | mask sweep + warp at inflate | HIGH (math direct) |
| + MXLZ renderer | −0.06 to −0.09 | sensitivity sweep + mixed-precision | MEDIUM (arch_header guard added) |
| + engineered corrections | −0.05 to −0.10 SegNet | gradient-directed pixels | LOW (no auth validation yet) |
| + winner_v2 noise | −0.10 to −0.20 SegNet | beneficial_quant_noise | SPECULATIVE (zero empirical backing) |
| **Realistic best stacked** | **1.6–1.8** | baseline + #1+3 + MXLZ + corrections | MEDIUM |
| **Optimistic best stacked** | **1.3–1.6** | + speculative v2 noise | LOW |
| **Aggressive (training breakthrough required)** | **<1.0** | requires `100·seg + √(10·pose) < (1.0 − rate_term)`. At rate=0.376 (current full-res masks) → distortion budget < 0.624. At rate=0.20 (after #1+3 mask reduction) → distortion budget < 0.80. Currently SegNet ≈ 0.116 contributes 11.6 alone — needs ~10× SegNet improvement, OR rate < 0.05 (basically impossible without removing scorer-required artifacts). | not yet measured |
| **To beat Quantizr (0.33)** | needs SegNet ~0.001, PoseNet ~0.001, rate ~0.15 | new arch family or major training breakthrough | NOT achievable with current architectures in 8 days |

> **R27 reviewer math:** earlier "0.8–1.4 realistic" was 4× the stated deltas. Sum check from 2.01: −0.18 (#1+3) − 0.09 (MXLZ) − 0.10 (corrections) = 1.64 floor. The −0.20 speculative v2 only reaches 1.44 if it actually delivers (no auth evidence).

## Non-Negotiable

- All experiments through `pipeline.py` invoked from `scripts/deploy_vastai.py launch()`. No ad-hoc shell scripts. No `nohup`. tmux only.
- `eval_roundtrip = True` everywhere.
- 3 consecutive auth evals within 0.01 before submission, AND 5 consecutive clean adversarial review passes.
- No ad-hoc scripts. Preflight check 6 enforces no `experiments/results/*/run_pipeline.sh` on remote.
- Destroy Vast.ai instances IMMEDIATELY after download — $24 hard cap.
- Every score labeled `[contest-compliant]` or `[unlimited-compute]`.
- Profile is the experiment definition. Any flag passed via launcher CLI must come from the profile dict.
- I4LZ weight compression DISABLED for non-default arch (padding_mode != zeros, use_dilation, use_zoom_flow) until I4LZ format gains an arch header.
