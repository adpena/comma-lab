# B1 229K PR95-faithful pilot — launch supplement (b1_229k_pilot_20260609T055851Z)

UTC: 2026-06-09T06:01:33Z · agent b1_baseline_launch · commit c2a863465d66bb1b998ca614fc643390a41b2026
Authority: `[macOS-MLX research-signal]` — score_claim=false, promotable=false,
ready_for_exact_eval_dispatch=false. ALL training numbers are MLX-advisory; the
SCORE is the B2 bridge exact eval on byte-closed archive.zip (NEVER MLX).

Companion of the gated manifest `.omx/research/b1_launch_manifest_b1_229k_pilot_20260609T055851Z.json`
(manifest_complete_and_self_consistent=True). This supplement records the P3
fields the manifest builder predates (launch command, batch schedule, diagnostics
cadence) + the launch-unblock findings.

## THE LAUNCH-UNBLOCK (P1) — what was actually blocking, and the fix

Two real blockers were diagnosed (NOT the prerequisites a prior agent guessed —
those were for the WRONG launcher, the compact runner `--execute-family hi_nerv`):

1. **Direct `--full` was structurally REFUSED.** The trainer
   `experiments/train_substrate_hi_nerv_mlx_local.py` exists with every needed
   flag and exports a backend-only archive end-to-end, BUT `_full_main` is gated
   by `_direct_trainer_canonicalization_contract` which hardcodes
   `trainer_launch_allowed=False` for ALL direct `--full` launches (it routes to
   the compact runner for queue-owned launch custody). The COMPACT RUNNER's
   `hi_nerv` family, however, builds `decoder_channels=tuple([c]*7)` (UNIFORM
   taper) + ratio-locked latents (line ~12802), so it STRUCTURALLY CANNOT express
   the 229K-parity asymmetric taper (36,30,23,17,14,11,8) with latents 16/20/24.
   => No existing path could launch the exact 228,903-param config.
   **FIX**: added `--allow-direct-research-full-launch` opt-out to the direct
   trainer. It flips `trainer_launch_allowed=True` for an HONEST research-signal
   launch (role tagged `explicit_operator_research_launch_macos_mlx_research_signal`)
   and RECORDS (not drops) the queue-ownership blockers under
   `research_launch_acknowledged_blockers`. The full PR95 production control
   contract stays ENFORCED. Operator-authorized per 2026-06-09 "Full send now:
   229K full curriculum".

2. **`--full` hardcoded epochs >= 29650** (`CANONICAL_PR95_FULL_EPOCHS`), making
   a reduced MVP pilot impossible. **FIX**: `--research-curriculum-total-epochs`
   (opt-out only) relaxes the epoch floor to an explicit reduced budget; relaxed
   blockers recorded under `research_relaxed_blockers` (acknowledged-not-dropped).

3. **Config trap (the predecessor's incomplete resume_command)**: the manifest
   builder's default resume command passed only `--decoder-channels` — which
   yields 159,433 params (the `hi_nerv_local_tiny` row has latents 8/10/12,
   embed 32). To hit EXACTLY 228,903 the launch MUST also override
   `--latent-dim-coarse 16 --latent-dim-mid 20 --latent-dim-fine 24 --embed-dim 64`
   AND `--num-pairs 600` (latents scale with pair count). VERIFIED via the
   trainer config path: 228,903 exact.

4. **Throughput-fix reachability gap**: `diagnostics_every_n_steps` (the
   ~1.65-1.78x cadence speedup) lived on the adapter but was NOT threaded through
   `run_mlx_score_aware_full_main`. **FIX**: threaded harness->adapter +
   `--diagnostics-every-n-steps` trainer flag + 2 regression tests (harness
   forwards the kwarg). Math-preserving (parity 0.0 per the throughput memo).

## End-to-end validation (BEFORE the real launch)

A real-teacher 8-pair/16-epoch full-path smoke confirmed the unblock:
loss 380.93 (ep0) -> 22.51 (ep15) (~17x decrease, monotone), backend-only
archive.zip exported (single member `x`, 196,741 bytes, real sha), EMA shadow +
checkpoints + telemetry + receiver-cache-quality probes all produced. The PR95
control contract PASSED; the canonicalization refusal was gone.

## The launched pilot (P4)

- Run id: `b1_229k_pilot_20260609T055851Z`
- SSD run dir: `/Volumes/VertigoDataTier/pact/b1_229k_pilot_20260609T055851Z`
- Driver (resumable, with heartbeat): `/Volumes/VertigoDataTier/pact/b1_229k_pilot_20260609T055851Z/launch_b1_pilot.sh`
- Heartbeat: `.omx/tmp/heartbeat_b1_b1_229k_pilot_20260609T055851Z.log` (60s cadence)
- Train log: `/Volumes/VertigoDataTier/pact/b1_229k_pilot_20260609T055851Z/train.log`
- Checkpoints: `/Volumes/VertigoDataTier/pact/b1_229k_pilot_20260609T055851Z/checkpoints` (every 250 epochs; resumable; crash never
  loses > 1 stage)
- EMA shadow archive selection: `/Volumes/VertigoDataTier/pact/b1_229k_pilot_20260609T055851Z/ema_archive_selection/`

### Config (exact 229K-parity, fail-closed verified)
- decoder_channels=(36,30,23,17,14,11,8), latents 16/20/24, embed 64, num_pairs=600
  => 228,903 params (== B1_PARAM_COUNT_EXACT; <= 228,958 parity).
- Muon partition (selective, NOT pure Muon): 191,104 Muon (10 tensors) /
  37,799 AdamW (17 tensors). V1's split differs from PR95-source 177,156/51,802
  because V1 is a different HiNeRV vehicle (documented in launch_manifest.py).
- 8-stage PR95-faithful curriculum, muon_active stages 1-7 = FALSE, stage 8 = TRUE
  (faithful_stage8_only). Reduced to 3000 epochs (MVP-first; exercises all stages
  incl. stage-8 Muon; ~3.6h at ~4.3 s/epoch).
- EMA decay 0.997 (saved as inference shadow). eval_roundtrip_ste ON. PR95 source
  weight amplification ON. Coder-aware QAT + C1a entropy ON. All scorer-input
  guards + posenet YUV6 geometry/temporal tethers ON. scorer-space step guard ON.
- diagnostics_every_n_steps=50 (throughput-fix; ~1.7x speedup, math-preserving).
- Batch schedule per `b1_large_batch_timing_sweep.v1`: early_search=16 (moves
  proxy most/epoch), qat_final=64 (stable QAT/final). This pilot uses 16
  throughout for max early proxy movement; a full-budget continuation can step to
  64 for the QAT/final stages.
- sidecar_export_enabled=FALSE (backend-only archive; single ZIP member `x`;
  auto strip available via strip_target_region_action_from_archive_payload if a
  target-region sidecar is ever enabled).

### RESUME command (full config, fixes predecessor's incomplete cmd)
```
.venv/bin/python experiments/train_substrate_hi_nerv_mlx_local.py --full --allow-direct-research-full-launch --research-curriculum-total-epochs 3000 --pr95-faithful-curriculum --pr95-muon-policy faithful_stage8_only --pr95-stage-source-weight-amplification --decoder-channels 36,30,23,17,14,11,8 --latent-dim-coarse 16 --latent-dim-mid 20 --latent-dim-fine 24 --embed-dim 64 --num-pairs 600 --batch-pairs 16 --ema-decay 0.997 --ema-archive-selection --diagnostics-every-n-steps 50 --distillation-weight 1.0 --segnet-distillation-objective boundary_argmax_hinge --segnet-direct-live-distillation-weight 1.0 --segnet-direct-live-class-histogram-weight 1.0 --pose-distillation-weight 1.0 --pose-direct-live-distillation-weight 1.0 --eval-roundtrip-ste --coder-qat --coder-qat-c1a-entropy-weight 1.0e-4 --hard-byte-ceiling 300000 --scorer-space-step-guard --scorer-input-distribution-guard-weight 1.0 --scorer-input-contrast-floor-weight 1.0 --scorer-input-shape-tether-weight 1.0 --posenet-yuv6-geometry-tether-weight 1.0 --posenet-temporal-signal-floor-weight 1.0 --checkpoint-selection-metric-key total --checkpoint-selection-metric-required --post-export-receiver-cache-quality-gate --upstream-dir upstream --checkpoint-dir /Volumes/VertigoDataTier/pact/b1_229k_pilot_20260609T055851Z/checkpoints --checkpoint-interval-epochs 250 --output-dir /Volumes/VertigoDataTier/pact/b1_229k_pilot_20260609T055851Z --resume-from-checkpoint /Volumes/VertigoDataTier/pact/b1_229k_pilot_20260609T055851Z/checkpoints
```

### EXACT-EVAL handoff (B2-owned; DO NOT run from B1)
```
.venv/bin/python tools/run_hi_nerv_backend_only_b2_exact_eval.py --archive /Volumes/VertigoDataTier/pact/b1_229k_pilot_20260609T055851Z/<best_ema_archive>/archive.zip --device cpu --json-out .omx/research/b1_first_exact_score_<utc>.json  # B2 BRIDGE (DO NOT run from B1; B2-owned): backend-only archive.zip -> inflate.sh -> upstream/evaluate.py CPU (Linux x86_64 authoritative; macOS advisory) -> first exact score vs 0.19199 frontier; --device cuda for T4 axis
```
The pilot EXPORTS a backend-only archive.zip at the end (best EMA shadow). That
archive -> the B2 bridge `tools/run_hi_nerv_backend_only_b2_exact_eval.py`
(--archive ... --device cpu) -> upstream/evaluate.py CPU (Linux x86_64
authoritative; macOS advisory) -> the FIRST exact backend-only HiNeRV score vs
the 0.19199 [contest-CPU] frontier.

## NO-FAKE attestation
Real contest pairs (upstream/videos/0.mkv); real SegNet/PoseNet safetensors
teachers (upstream/models/); real PR95FaithfulCurriculumFactory + real
partition_pr95_mlx_parameter_names (param count BUILT from the real MLX model,
not hand-derived). The reduced-epoch pilot is honestly tagged: it will NOT reach
gold (compressed budget); its job is the FIRST exact-score signal + full-pipeline
validation (train -> export -> backend-only archive). If promising at B2, the full
29650-epoch continuation follows (resumable from the same checkpoints).

## 6-hook wire-in (Catalog #125)
1. Sensitivity-map — N/A (training launch, not a per-axis byte-savings lane).
2. Pareto constraint — N/A (no archive-byte polytope change at launch).
3. Bit-allocator hook — N/A (no per-tensor importance change at launch).
4. Cathedral autopilot dispatch hook — N/A (advisory MLX training; not
   archive-deployable until B2 exact eval).
5. Continual-learning posterior — this supplement + the gated manifest +
   per-epoch telemetry.jsonl are the empirical anchors; the EMA-selected archive
   feeds the B2 bridge.
6. Probe-disambiguator — N/A (no competing interpretations; the contract
   verdicts are deterministic).
