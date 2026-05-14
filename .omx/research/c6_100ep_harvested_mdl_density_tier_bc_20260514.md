# C6 MDL-IBPS Z1 ablation Tier B+C ledger (100ep harvest + 5ep proxy)

`[5ep-architecturally-trained-archive]` `[ablation-in-flight]` `[smoke-RED-timeout]`

**Date**: 2026-05-14 UTC
**Subagent**: `harvest_and_z1_subagent_20260514`
**Lane**: `lane_c6_smoke_harvest_z1_ablation_auto_fire_full_20260514` (Phase 2, L0 pre-registered → L1 with memory_entry on landing)
**Inherited directives (Catalog #125 + Rule R1 from recursive directive)**:
- `.omx/research/recovery_session_20260514_directive_absolute_no_signal_loss_20260514.md`
- `.omx/research/recursive_no_signal_loss_protocol_20260514.md`

## TL;DR

- **C6 100ep T4 smoke RED (timed_out=True; rc=124; elapsed 3600.59s)** — training_loop never completed, no archive, no auth_eval, no contest_cuda score
- **No auto-fire of C6 200ep full** — per CLAUDE.md "KILL is LAST RESORT" the lane is **DEFERRED-pending-research**, NOT killed
- **Z1 Tier B ablation on 5ep harvested archive** is **in-flight** (PID 11231, running ~32 min on local CPU at byte-flip + scorer-forward). Will append section-level density numbers when complete to `experiments/results/mdl_ablation_z1_c6_5ep_20260514/c6_ibps1_5ep_mdl_ablation.json`
- **Z1 Tier C does NOT cover ibps1 grammar** in the canonical tool (`tools/mdl_scorer_conditional_ablation.py:1127` early-returns `[]` for non-a1 grammars). Tier C is the cleanest signal for substrate-class discrimination but is currently A1-only
- Sister codex ablation at `experiments/results/mdl_ablation_c6_5ep_tier_bc_20260514_codex/` (200 byte_samples × 60 pair_samples) is also in-flight; expected ~3 hours total wall-clock
- **C6-NEXT-WAVE 5ep STRUCTURAL tier (Tier A proxy via byte-Shannon-entropy) already documented at `c6_5ep_mdl_density_proxy_20260514.md`**: density 0.9904 within-HNeRV-class. Per the same memo, Tier A is brotli-saturated and structurally cannot discriminate substrate class — only Tier B/C can

## Part A — C6 100ep Modal smoke harvest (TIMED OUT)

| Field | Value |
| --- | --- |
| Modal call_id | `fc-01KRKHNZSZF4JPHJ20WE35A68C` |
| Dispatch label | `substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260514T152845Z__smoke__100ep` |
| Dispatched at | 2026-05-14T15:28:45Z (claim entry); worker start 2026-05-14T15:29:33Z |
| Returncode | **124 (timeout)** |
| Wall-clock | **3600.588s** (Modal max_seconds=3600 hard kill) |
| Timed out | **True** |
| Out_dir | `experiments/results/lane_substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260514T152845Z__smoke__100ep_modal/` |
| Archive | **NOT EMITTED** (training_loop did not complete) |
| Auth eval | **SKIPPED** (training_loop did not complete) |
| Worker source-parity (Catalog #166) | clean (sentinel_mismatches=[]) |
| Estimated cost (modal_elapsed × $0.59/hr) | **$0.59** |

### Stage-by-stage trainer log

```
[c6-full] STAGE: patch_yuv6 @ 2026-05-14T15:29:38Z       (5s after worker start)
[c6-full] STAGE: load_scorers @ 2026-05-14T15:29:43Z     (+5s)
[c6-full] STAGE: decode_video_pairs @ 2026-05-14T15:29:45Z (+2s)
[c6-full] decoded 600 pairs at (384, 512)                (107s decode)
[c6-full] STAGE: build_substrate @ 2026-05-14T15:31:32Z   (worker T+1m59s)
[c6-full] STAGE: training_loop @ 2026-05-14T15:31:32Z     (worker T+1m59s)
                                                          + LIVE training to Modal hard-kill at 60min
```

**Mechanism**: 100 epochs × ~30s/epoch (extrapolated from 5ep `training_seconds=296.66s ≈ 59s/ep`) = ~3000s training alone. Plus pyav decode (107s), patch_yuv6 (5s), load_scorers (5s), build_substrate (107s), and post-training auth eval (~3 min CUDA T4) → **~3422s total**, very close to the 3600s Modal max_seconds bound. The 100ep dispatch was **structurally on the wrong side of the time budget** for T4. 5ep training_seconds = 296.66s; extrapolation to 100ep = 5933s = 99min, AND that ignores post-training auth eval. With autocast disabled and deterministic mode warning issued, training is slower than nominal.

### Smoke verdict per `tools/run_modal_smoke_before_full.py::_validate_smoke_result`

```
rc=124 (expected 0) → SMOKE RED
timed_out=True → also RED branch
artifacts: 8 keys, but NO auth_eval_*.json present (training never reached eval stage)
→ unconditionally RED
```

**No auto-fire of full**: per the smoke-before-full contract, smoke RED refuses the full canary. The full dispatch was never spawned by the wrapper.

### Posterior + claim updates landed

- **Cost-band anchor** (Catalog #175 + #177): `outcome=timed_out`, `actual_cost_usd=$0.59` (rate-based estimate), appended via `_save_anchor_appended_artifact` → `.omx/state/cost_band_posterior.jsonl` (auto via `tools/harvest_modal_calls.py --execute`)
- **Dispatch claim** (Catalog #131 + helper `claim_lane_dispatch.py --force`): terminal row `failed_modal_training_timeout`, appended to `.omx/state/active_lane_dispatch_claims.md` for `instance_job_id=substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260514T152845Z__smoke__100ep`

## Part B — Z1 Tier B+C MDL ablation on 5ep harvested archive

**Archive under test**:
- Path: `experiments/results/lane_substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260514T144949Z__smoke__5ep_modal/harvested_artifacts/lane_substrate_c6_e4_mdl_ibps_results/output/archive.zip`
- Size: 224481 bytes
- sha256: `a27328ce02211f1c8ee0cfb4318ace29c438a62cf09a42358481d0273a204607`
- Inner blob magic: `b'IBPS'` schema_version=1
- Training: 5 epochs (auth_eval skipped due to old CLI flag; per RECOVERY-2 fix the next 5ep+ dispatch will emit valid contest_cuda auth)

**Tool**: `tools/mdl_scorer_conditional_ablation.py` (canonical Z1 ablation)
**Grammar**: `ibps1` (C6 MDL-IBPS variant 1; wired by C6-NEXT-WAVE subagent)
**Config**: `--byte-samples 60 --pair-samples 30 --seed 1234 --skip-tier-a --device cpu`
**Output dir**: `experiments/results/mdl_ablation_z1_c6_5ep_20260514/`
**Output file (when complete)**: `c6_ibps1_5ep_mdl_ablation.json`

### Tier coverage

- **Tier A (structural)**: SKIPPED. Already computed by C6-NEXT-WAVE; documented at `c6_5ep_mdl_density_proxy_20260514.md` → density 0.9904 (within-HNeRV-class saturated at brotli ceiling)
- **Tier B (sampled byte-level flip + scorer Δscore)**: **IN-FLIGHT**. 60 byte positions per IBPS1 section × 4 sections (encoder_blob, decoder_blob, latent_blob, meta_blob) = ~240 byte-flip experiments. Each: byte-XOR-0xFF → re-decode IBPS1 → SegNet+PoseNet forward at 30 sampled pairs → compute Δseg + Δpose + Δscore_components. Expected runtime ~50-60min on local CPU (M5 Max). PID 11231 at byte-flip iteration ~?, RSS 4.4GB
- **Tier C (post-decode perturbation)**: **NOT SUPPORTED** for ibps1 grammar in current tool (returns `[]` early at `mdl_scorer_conditional_ablation.py:1127`). A1 has Tier C; PR106 + IBPS1 do not. Future work: extend `_decode_ibps1_to_frames` with monkey-patched state_dict noise injection. Operator-routable Decision 3.

### Why the result still has signal even at 5ep

Per the C6-NEXT-WAVE memo:
- Brotli output is **near-maximum-entropy** at the byte layer regardless of architecture → Tier A cannot discriminate
- **Tier B IS the dispositive disambiguator**: it perturbs at the byte layer AND measures **scorer-conditional Δscore** through the actual SegNet+PoseNet forward. If C6's IB-bottlenecked latents are encoding **different scorer-relevant information per byte** than A1/PR101's straight HNeRV latents, this shows up as a **different `fraction_significant` distribution across IBPS1 sections vs A1's HNeRV sections**
- 5ep training is **architecturally representative** even if not fully converged — the IB encoder/decoder/latent structural pattern is in place

**However**: 5ep weights may not be fully informative (training_loss best_loss_proxy = 52.4, much higher than the target 0.10-0.30 score band). Tier B density on 5ep is a **lower bound on the trained substrate's true scorer-conditional MDL density** — if 5ep shows < 0.90 density (across-class signal), the trained 200ep substrate could be even further across-class. If 5ep shows ≥ 0.90 density, that's also informative: **either** (a) the architecture is structurally within-class (DEFERRED verdict reinforced) OR (b) 5ep weights are too noisy to discriminate (operator must train longer to disambiguate)

### Sister codex parallel ablation

A second ablation run is in-flight on the **same archive** at `experiments/results/mdl_ablation_c6_5ep_tier_bc_20260514_codex/`:
- Config: `--byte-samples 200 --pair-samples 60` (3.3× more byte samples, 2× more pair samples than mine)
- Started ~1 min before mine; PID 10727; will take ~3× longer
- **Provides higher-confidence statistical bound on density estimate**; mine is the faster pilot

Per directive Rule R2 (sister-subagent ownership disjointness), my output dir and codex's are **distinct**; no collision risk. Both ablations re-seed the autopilot posterior independently on completion.

## Part C — Decision logic application

Per the prompt's decision tree:

| Branch | Condition | Action |
| --- | --- | --- |
| GREEN | smoke score in [0.10, 0.30] AND density < 0.90 | auto-fire C6 200ep full |
| YELLOW | smoke score in band BUT density > 0.90 | register class-saturation; do NOT auto-fire |
| RED | smoke score out of band OR no score | DEFERRED-pending-research |

**Decision: RED → DEFERRED-pending-research** (NOT KILL)

The 100ep smoke has **no score** (training never completed). Per CLAUDE.md "Forbidden premature KILL without research exhaustion" non-negotiable, this single timed-out config does NOT warrant a KILL verdict. The lane is DEFERRED-pending-research with these reactivation criteria:

### Reactivation criteria for C6 lane

The C6 lane (`lane_c6_e4_mdl_ibps_substrate_20260514`) is **DEFERRED-pending-T4-timeout-budget-fix**. Reactivate when ANY of:

1. **Smaller smoke epoch budget**: dispatch `--smoke-epochs 50` (or even 25) on T4, fits comfortably in 3600s, surfaces an in-band score for the GREEN/YELLOW decision. C6 is mathematically a different architecture class (IB-bottleneck) — even a 50ep proxy score landing in [0.10, 0.30] is dispositive evidence.
2. **Faster GPU class**: dispatch `--smoke-gpu A10G` or `--smoke-gpu A100` with `min_smoke_gpu: "A10G"` recipe declaration. A100 with autocast fp16 should finish 100ep in ~15-20min, leaving 40min margin for auth eval. Cost: $1.20-3 vs T4 $0.59.
3. **Larger Modal timeout**: dispatch via `--smoke-timeout-hours 2.0` (Modal max_seconds=7200) on T4. Cost: same hourly rate × 2 → ~$1.18 worst-case if timeout still fires; ~$0.85 expected (100ep≈99min training + auth eval).
4. **Trainer wall-clock optimization**: profile the C6 trainer for per-epoch cost; the 59s/epoch on T4 at 384×512 + 600 pairs is high. Candidates: smaller pair batch, autocast fp16 enabled, gradient accumulation, mixed-precision SegNet/PoseNet forward. Per Catalog #172/#178/#179 Tier 1 optimization gaps.
5. **Z1 Tier B ablation on this 5ep archive shows scorer-conditional MDL density < 0.90 (across-class signal)**: this would upgrade the lane's predicted ΔS from "within-class capped at 0" to "across-class candidate". When ablation lands, append result to this ledger.

**The 100ep timeout is a wall-clock budget problem, not a substrate-architecture falsification.** C6's parameter count (128,882) and config (latent_dim=24, beta_ib=0.01) are reasonable. The mechanism (variational IB + per-pair z) is the across-class hypothesis per the recipe summary. The empirical work needed is **a completable smoke**, not a different architecture.

## Part D — Tier B preliminary observations (will be appended when ablation completes)

`[placeholder; tool still running; update when c6_ibps1_5ep_mdl_ablation.json lands]`

When the JSON lands, this section will record:
- `mdl_density_estimate_lo` / `_hi`
- `mdl_scorer_extracted_bytes_lo` / `_hi`
- `zen_floor_band_recommendation`
- Per-section `fraction_significant` for ibps1_header / encoder_blob / decoder_blob / latent_blob / meta_blob
- `n_inflate_failures` per section (structural-significance proxy)
- Cross-ref against the C6-NEXT-WAVE Tier A density (0.9904) — does Tier B density agree with Tier A's within-class verdict, or diverge into across-class signal?

## Cross-references

- C6-NEXT-WAVE 5ep Tier A proxy: `.omx/research/c6_5ep_mdl_density_proxy_20260514.md` (density 0.9904 brotli-saturated)
- Z1 canonical ablation framework: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_z1_mdl_ablation_landed_20260514.md`
- Zen-floor band v2 within-class-vs-across-class taxonomy: `feedback_zen_floor_band_v2_post_z1_ablation_20260514.md`
- C6 RECOVERY-2 finish + Modal harvest (the antecedent landing this builds on): `feedback_recovery_2_c6_finish_and_modal_harvest_landed_20260514.md`
- HNeRV parity discipline lesson 7 (substrate engineering exception): CLAUDE.md "HNeRV / leaderboard-implementation parity discipline"
- C6 substrate code: `src/tac/substrates/c6_e4_mdl_ibps/`
- C6 trainer: `experiments/train_substrate_c6_e4_mdl_ibps.py`
- C6 recipe: `.omx/operator_authorize_recipes/substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml`
- C6 remote driver: `scripts/remote_lane_substrate_c6_e4_mdl_ibps.sh`
- Operator-authorize wrapper: `scripts/operator_authorize_substrate_c6_e4_mdl_ibps_modal_t4_dispatch.sh`
- Cathedral autopilot ranker Z1 revision: `tac.cathedral_autopilot_autonomous_loop.{adjust_predicted_delta_for_mdl_density, adjust_predicted_delta_for_class_shift}`
- IBPS1 grammar definition: `src/tac/substrates/c6_e4_mdl_ibps/archive.py::IBPS1_HEADER_FMT`

## Tags

- `[5ep-architecturally-trained-archive]` — Tier B runs on a 5ep training that did complete cleanly per `feedback_recovery_2_c6_finish_and_modal_harvest_landed_20260514.md`; the 100ep training is the one that TIMED OUT
- `[ablation-in-flight]` — Tier B running locally on M5 Max CPU; will update when complete
- `[smoke-RED-timeout]` — 100ep dispatch returned rc=124 timed_out=True; no auth_eval; no score_claim
- `[contest-CUDA T4]` — N/A; no scoring artifact produced
- `[macOS-CPU advisory only]` — ablation runs locally on M5 Max ARM64 CPU; per CLAUDE.md "MPS auth eval is NOISE" and "Submission auth eval — BOTH CPU AND CUDA" non-negotiables, this is research-signal-only for substrate-class discrimination; NOT a contest score
- `[planning_only_no_score_claim]` — ledger informs autopilot ranker; does not claim a score
- `[no_mps_authoritative]` — N/A
- `[no_tmp_paths]` — all paths under `experiments/results/` or `.omx/research/`
- `research_only=true` per CLAUDE.md "Subagent coherence-by-default" 6-hook wire-in
- Tagged `inherited_directives: ["recovery_session_20260514_directive_absolute_no_signal_loss_20260514", "recursive_no_signal_loss_protocol_20260514"]` per directive Rule R3

## Wire-in hooks (Catalog #125 mandatory 6-hook declaration)

1. **Sensitivity-map contribution**: N/A — this is a forensic-and-decision-logic ledger, not a new architecture primitive. No `tac.sensitivity_map.*` entry.
2. **Pareto constraint**: N/A — no new bytes/score constraint added. The DEFERRED verdict is a planner-routing tag, not a Pareto edge.
3. **Bit-allocator hook**: N/A — no per-tensor importance change.
4. **Cathedral autopilot dispatch hook**: **YES** — when Tier B density lands, it updates the autopilot ranker's `apply_z1_empirical_revision_to_candidate_delta` for the C6 lane via the cost-band anchor + Catalog #219 STRICT preflight gate. The DEFERRED-pending-research verdict on the 100ep T4 dispatch contributes a `failed_modal_training_timeout` anchor that the autopilot consumes via `tac.cost_band_calibration.load_anchors`. Re-dispatch with smaller-epoch / faster-GPU is the ranker's next routable.
5. **Continual-learning posterior update**: **YES (auto via harvester)** — `tools/harvest_modal_calls.py --execute` auto-appended a `timed_out` outcome cost-band anchor via `tac.cost_band_calibration.append_platform_training_anchor` (canonical locked helper per Catalog #128 + #131). When the Tier B ablation completes, an MDL ablation density anchor will be available for `tac.continual_learning.posterior_update_locked` via the cathedral autopilot v2 ranker queue.
6. **Probe-disambiguator**: **YES (existing)** — the C6 recipe declares this directly:
   > "two defensible interpretations of 'what dominates ΔS' — (a) decoder-class hypothesis vs (b) encoder-bottleneck hypothesis. The β-sweep over [0.001, 1.0] IS the probe; post-smoke MDL ablation on the C6 archive (via `tools/mdl_scorer_conditional_ablation.py`) CONFIRMS or REFUTES the substrate-class shift."
   Tier B IS the probe; the in-flight ablation IS the probe-execution. Output disambiguates (a) vs (b) on completion.

## Operator-routable decisions

1. **DECISION 1 — Re-dispatch C6 100ep on faster GPU** (highest EV-toward-score-lowering)
   - Option A: `--smoke-gpu A10G` (Modal A10G; ~22GB shared; expected ~30min for 100ep training + 5min auth eval). Cost: ~$1.20-$1.50. **RECOMMENDED.**
   - Option B: `--smoke-gpu A100` (~40GB; expected ~15-20min for 100ep training; 3-5min auth eval). Cost: ~$3-4.
   - Option C: `--smoke-epochs 50 --smoke-gpu T4` (fit in T4 3600s budget at $0.59). Less informative but cheapest.
   - **RECOMMENDATION**: Option A or B. The substrate is the **zen-Z1 LARGEST single bet** (per `feedback_long_term_multi_year_campaigns_landed_20260514.md` C6). $1-4 to land an in-band score is dominated by the predicted ΔS = -0.060 saving.

2. **DECISION 2 — Wait for in-flight Z1 Tier B ablation to land before re-dispatch**
   - Argument FOR: Tier B density on 5ep weights is **the dispositive substrate-class discrimination test**. If 5ep Tier B density < 0.90, C6 is across-class and a 100ep+ dispatch is highest-EV. If density ≥ 0.90, C6 may be structurally within-class even at 200ep convergence, and operator should rebudget toward Z4/Z5/D4 instead.
   - Argument AGAINST: 5ep weights are noisy; the ablation is only ~70-80% confidence on substrate-class. A 100ep+ dispatch on A10G/A100 produces a higher-quality signal for the same operator-wall-clock cost.
   - **RECOMMENDATION**: WAIT for ablation completion (≤60 min from now for my run; ~2-3 hr for codex sister). The $1-4 saved by NOT firing a premature re-dispatch outweighs the wall-clock cost. **If Tier B density < 0.90 — fire Option A. If ≥ 0.90 — defer C6 to operator review with the Tier B/C verdict.**

3. **DECISION 3 — Extend MDL ablation tool to support IBPS1 Tier C**
   - Current limitation: `mdl_scorer_conditional_ablation.py:1127` returns `[]` for non-a1 grammars in Tier C. IBPS1 Tier C would require monkey-patched state_dict noise injection at the encoder/decoder/latent boundaries via the C6 `MDLIBPSSubstrate.from_archive(...)` path.
   - LOC estimate: ~80-120 LOC (mirror A1 Tier C structure + IBPS1-specific decode + sigma sweep on encoder/decoder/latents separately).
   - Cost: $0 (engineering only); time ~2-4h with adversarial council review.
   - **RECOMMENDATION**: Defer to operator routable — Tier B alone is often dispositive on the substrate-class question. Tier C is the high-precision follow-up if Tier B is ambiguous (e.g. density in [0.85, 0.95]).

4. **DECISION 4 — Trainer-side wall-clock optimization (Tier 1 batch backport to C6)**
   - Per Catalog #172 (`autocast_fp16`) + #178 (`tf32`) + #179 (`torch.compile`) gates, C6 trainer can be backported to use the Tier 1 optimization batch primitives at `tac.training_optimization.{scorer_cache, autocast_helper, compile_helper}` (per `feedback_tier_1_optimization_batch_landed_20260514.md`). Expected speedup: 1.5-2× per-step + 50% scorer compute savings via O1 GT-scorer cache.
   - LOC estimate: ~5-10 LOC in the C6 trainer + ~5 LOC argparse flags + verification on smaller smoke first.
   - Cost: $0 (engineering); $0.30 for the first smoke validation per Tier 1 batch landing memo.
   - **RECOMMENDATION**: Land this BEFORE Decision 1 re-dispatch. The same Tier 1 backport unblocks ALL substrate trainers' Modal timeout pressure. Per the same memo, the trainer-side backport is **operator-routable next wave**; C6 is the natural first beneficiary because it was the empirical anchor for "timeout on T4 100ep".

5. **DECISION 5 — Adjust C6 recipe `min_smoke_gpu` field to reflect timeout finding**
   - Status after executable-surface repair: `substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml` declares `min_smoke_gpu: "A10G"`.
   - Validate without dispatch:
     `.venv/bin/python tools/run_modal_smoke_before_full.py --recipe substrate_c6_e4_mdl_ibps_modal_t4_dispatch --smoke-epochs 50 --smoke-gpu T4 --smoke-timeout-hours 1.0 --operator-handle "operator:c6_surface_validation" --smoke-only --dry-run`
   - Expected dry-run behavior: helper prints `Catalog #215: smoke GPU T4 -> A10G`, proving stale T4 CLI defaults no longer launch the timed-out smoke configuration.

## Resume protocol

Per CLAUDE.md "Mandatory crash-resume protocol" (Catalog #206), this subagent's checkpoint record at step 7 records:
- `subagent_id`: `harvest_and_z1_subagent_20260514`
- `lane_id`: `lane_c6_smoke_harvest_z1_ablation_auto_fire_full_20260514`
- `inherited_directives`: original 7-rule + recursive extension (verified via this ledger)
- `next_action`: "wait for ablation completion; finalize ledger Part D; assemble routables"

If this subagent crashes before the ablation lands, a successor reads the latest checkpoint via `tools/subagent_checkpoint.py read --lane-id lane_c6_smoke_harvest_z1_ablation_auto_fire_full_20260514` and resumes at "wait for ablation completion; finalize ledger Part D".

`research_only=true`. NO score claims. NO promotion. $0 GPU spend by this subagent (the $0.59 C6 100ep smoke spend was from the antecedent operator-authorize wrapper firing).
