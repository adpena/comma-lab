# DP1 Pretrained Driving Prior - Forensic Audit And Roadmap

**Date:** 2026-05-15
**Author:** Codex
**Lane family:** pretrained driving prior / DP1
**Primary lane ids:** `driving_prior_pretrained_renderer_2032`, `lane_pretrained_driving_prior_lane_scaffold_20260513`, `lane_pretrained_driving_prior_phase_2_20260514`, `lane_dp1_phase_2_hardening_v2_20260514`
**Conclusion:** DP1 is not dead and not falsified. It is implemented enough to smoke, parse, probe, run Tier-C advisory analysis, apply its codebook at inflate time, and enter a controlled full-training path. It has not yet produced a trained driving prior, deployment packet with promotion-grade custody, or legitimate contest-CPU / contest-CUDA score.

## Executive Status

DP1 currently has three evidence classes:

1. **Planning/readiness evidence only.** `reports/cooperative_receiver/driving_prior_readiness.json` reports `training_started=false`, `dispatch_attempted=false`, `gpu_used=false`, `archive_materialized=false`, `contest_cuda_auth_eval=false`, `ready_for_exact_eval_dispatch=false`, `promotion_eligible=false`, `score_claim=false`, and `score_evidence_grade=invalid_no_score_planning_only`.
2. **Smoke/proxy archive evidence.** `experiments/results/dp1_smoke_v2_hardening/manifest.json` records a deterministic smoke archive ZIP of 12,032 bytes with `training_mode=smoke`, `evidence_grade=[proxy]`, `contest_cuda_eval_not_run`, `contest_cpu_eval_not_run`, `real_codebook_distillation_pending`, and `real_renderer_training_pending`.
3. **Tier-C real-scorer advisory evidence.** `experiments/results/tier_c_real_scorer_fourway_codex_execute1_dp1fix_20260514/dp1_smoke_tier_c_real_scorer.json` reports `mdl_tier_c_density_estimate=0.13258188436368462` and `mdl_tier_c_substrate_class_verdict=across_class`, but explicitly marks the result as `[real-scorer CPU Tier-C delta curves; pair-sampled; no score claim]`, `[macOS-CPU advisory only]`, not contest-CUDA, not full contest-CPU, and not promotion authority.

No trained Comma2k19 / BDD100K / Waymo prior has landed. No DP1 full-training run has been harvested. No DP1 deployment packet has passed submission compliance. No paired same-archive contest-CPU and contest-CUDA auth eval exists.

## Forensic Classification

**Current classification:** `untrained_unpromoted_promising_substrate`.

**Dead/falsified classification is forbidden at this time** because every negative conclusion would be confounded by missing engineering prerequisites:

- real dataset source not supplied or licensed for the run;
- real codebook distillation not executed;
- renderer training not executed;
- byte-closed archive and inflate runtime not promoted through compliance checks;
- no trained-archive no-op consumption proof that every charged section affects inflated frames;
- no paired contest-CPU / contest-CUDA auth eval on one archive/runtime;
- composition runtime missing for DP1 x A1, DP1 x YUCR, and DP1 x PR101/PR106 cells;
- Tier-C evidence is pair-sampled advisory signal, not rank/kill evidence;
- prior score benefit is expected to be small in contest-only mode because the official scorer already contains a driving prior.

DP1 may only be retired as a measured configuration after a result-review packet verifies custody, axis, runtime config, archive/runtime closure, formula recomputation, and terminal dispatch claim. DP1 as a family may only be killed after multiple trained, byte-closed, composition-aware exact-eval cells fail with no engineering/config blockers and with reactivation criteria recorded.

## Engineering Fixes Landed In This Pass

1. `tools/build_result_review_packet.py` now emits an `engineering_forensic_audit` block. Negative exact-CUDA results with missing runtime closure or missing terminal dispatch claim stay `indeterminate_engineering_or_config_blocker` instead of retiring a measured config.
2. `src/tac/preflight.py` now requires dead/falsified/exact-negative/rank-or-kill evidence rows to carry either an inline engineering forensic audit or a valid exact result-review packet. Rows with `engineering_or_config_bug_found=true` cannot simultaneously claim family/method/rank-or-kill status.
3. `src/tac/substrates/pretrained_driving_prior/inflate.py`, `prior_application.py`, and `score_aware_loss.py` now make the archived codebook score-affecting: inflate applies the deterministic soft-prior transform before residuals, and training scores the same post-prior path. `src/tac/tests/test_dp1_inflate_consumes_codebook.py` proves two archives differing only in codebook bytes produce different raw outputs.
4. `scripts/remote_lane_substrate_pretrained_driving_prior.sh` now threads the DP1 full-run controls for GTScorerCache, torch.compile, streaming/cache/chunking, memory caps, distillation caps, pair caps, validation cadence, CPU advisory flags, and auth-eval skip. Full Comma2k19 mode now fails closed unless an explicit dataset source is supplied.
5. `.omx/operator_authorize_recipes/substrate_pretrained_driving_prior_modal_t4_dispatch.yaml` no longer claims `_full_main` raises `NotImplementedError`. The recipe now says full path exists but remains gated by `DPP_RUN_FULL=1`, deterministic dataset source, runtime custody, no-op proof, and paired CPU/CUDA eval.
6. `src/tac/substrates/_shared/trainer_skeleton.py` now supports an explicit `allow_full_cpu=True` advisory exception, and DP1 wires `--full-cpu --advisory-cpu-explicitly-waived` through that path. This fixes the mismatch where DP1 advertised CPU advisory full training but the shared device gate refused it before training.
7. DP1 residual export now encodes signed int8 values as two's-complement bytes via `int(v) & 0xFF`, fixing the archive-build crash where negative int8 residuals were passed directly to `bytes()`.

Initial focused verification run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  src/tac/tests/test_build_result_review_packet.py \
  src/tac/tests/test_preflight_harden_2026_05_08_checks.py \
  src/tac/tests/test_preflight_meta_bugs.py::TestEvidenceFalsificationScopeGuard
```

Result: `36 passed`.

Expanded focused verification after the codebook-consumption fix:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  src/tac/tests/test_build_result_review_packet.py \
  src/tac/tests/test_preflight_harden_2026_05_08_checks.py \
  src/tac/tests/test_preflight_meta_bugs.py::TestEvidenceFalsificationScopeGuard \
  src/tac/tests/test_dp1_remote_driver_contract.py \
  src/tac/tests/test_dp1_inflate_consumes_codebook.py \
  src/tac/tests/test_f3_backport_vqvae_pdp_wired.py
```

Result: `56 passed`.

Smoke after the prior-consumption fix:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python \
  experiments/train_substrate_pretrained_driving_prior.py \
  --smoke --device cpu \
  --output-dir experiments/results/dp1_forensic_smoke_20260515_codex_after_prior_fix
```

Result: smoke pack/parse roundtrip succeeded; archive ZIP and manifest were written.

Tiny full CPU advisory probe after full-CPU and residual-export fixes:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python \
  experiments/train_substrate_pretrained_driving_prior.py \
  --device cpu --full-cpu --advisory-cpu-explicitly-waived \
  --dataset-name synthetic_test \
  --epochs 1 --batch-size 1 --max-pairs 4 --val-pair-count 1 \
  --max-distillation-frames 16 --max-distillation-chunks 1 \
  --skip-auth-eval \
  --output-dir experiments/results/dp1_tiny_full_cpu_advisory_20260515_codex
```

Result: full path trained 1 epoch on real contest pairs with synthetic codebook, wrote `0.bin` (25,814 B), `archive.zip` (25,914 B), `manifest.json`, and `provenance.json`. This is `[proxy]` / CPU advisory / no score claim because auth eval was skipped and the codebook was synthetic.

## Outstanding Gaps

### G1 - Dataset source and licensing

The readiness artifact lists Comma2k19, BDD100K, and Waymo as optional external sources, with no downloads attempted and no local paths configured. The next real DP1 run must bind one explicit source:

- `DPP_COMMA2K19_CHUNKS_DIR` for operator-supplied local MIT Comma2k19 chunks;
- `DPP_CACHE_DIR` for explicit cache location if auto-fetch is approved;
- `DPP_USE_STREAMER=1` plus `DPP_STREAM_LOG_DIR` for streaming/log mode.

No dataset bytes should be committed. Every codebook must carry dataset provenance, license tags, source chunk ids, source SHA-256s when available, random seed, frame count, and distillation config.

### G2 - Real training not run

The trainer has `_full_main`, and a tiny synthetic CPU advisory full-run now proves the training/export path can execute. Real DP1 remains untrained because no Comma2k19/BDD/Waymo codebook training run has been harvested. Required first empirical training sequence:

1. repeat tiny full run with an explicit real dataset source and `--skip-auth-eval` to prove Comma2k19/cache/streamer ingestion;
2. same tiny run with exact auth eval on the intended CPU/CUDA axis if the runtime is byte-closed;
3. timing smoke on Modal T4/A100 with real dataset source to estimate seconds/epoch, RSS/VRAM, and cost;
4. converged full run only after the timing smoke establishes a rational budget.

### G3 - Byte-closed deployment packet missing

The full trainer can write `submission/inflate.sh`, `submission/inflate.py`, `archive.zip`, `0.bin`, `manifest.json`, and `provenance.json`, but no trained packet has been harvested. Before promotion:

- run `scripts/pre_submission_compliance_check.py --contest-final --strict ...`;
- record archive ZIP bytes/SHA-256 and payload member SHA-256;
- record runtime tree SHA and inflate script SHA;
- compare full-frame inflate output parity where source/candidate parity is claimed;
- prove no PoseNet/SegNet/scorer imports occur at inflate time;
- prove codebook, renderer, and residual sections are each output-affecting for the trained archive, not only for the unit fixture;
- preserve exact command, hardware, logs, and dispatch claim terminal row.

### G4 - Paired axis evidence missing

DP1 needs paired auth eval by default. For one archive/runtime, record:

- `[contest-CUDA]` exact eval on contest-compliant CUDA hardware;
- `[contest-CPU]` exact eval on contest-compliant Linux x86 CPU hardware if the public CPU axis is being considered;
- component deltas for PoseNet and SegNet;
- score formula recomputation from components and byte term;
- explicit output JSON paths whose device tokens match their contents.

MacOS-CPU and pair-sampled Tier-C remain advisory. Modal CPU may be valid only when the evaluator/runtime/hardware contract marks it as contest-CPU, not advisory CPU.

### G5 - Composition runtime not built

The DP1 composition API is a byte-level framing primitive, not yet a shipping composed inflate runtime. Required cells:

- DP1 x A1 first, because A1 has a verified baseline anchor;
- DP1 x YUCR second, because YUCR should expose scorer-blindspot / cost-map synergy;
- DP1 x PR101/PR106 only after public-frontier custody is exact and the base runtime is reproduced.

Each composition run must measure the interaction term:

```text
interaction = delta_score(DP1 x base) - delta_score(DP1 alone) - delta_score(base alone)
```

No composition claim is valid without one archive/runtime that actually combines outputs, not just concatenated bytes.

### G6 - Solver/autopilot hooks not score-bearing yet

Current solver wire-in is planning grade:

- sensitivity map contribution is planned from scorer penultimate saliency;
- Pareto constraint is non-binding until byte-closed archive and exact eval;
- bit allocator hook is planned after trained prior exists;
- cathedral autopilot dispatch is blocked by proxy-only evidence;
- continual-learning posterior update is disabled until empirical anchor;
- probe-disambiguator has hook targets but no exact anchor.

The first exact anchor must update the posterior with axis label, archive SHA, byte count, hardware, and component distances.

### G7 - Tier-C density is promising but not enough

The `0.13258188436368462` Tier-C density estimate is high signal for across-class behavior, not a score. It is only one pair, macOS CPU advisory, and rate term excluded. It can justify a timing smoke; it cannot justify promotion, retirement, or leaderboard claims.

### G8 - Performance and memory caps must be explicit

The remote full path now supports:

- `DPP_ENABLE_GT_SCORER_CACHE`;
- `DPP_ENABLE_TORCH_COMPILE`;
- `DPP_ENABLE_AUTOCAST_FP16`;
- `DPP_USE_STREAMER`;
- `DPP_STREAM_LOG_DIR`;
- `DPP_RAM_BUFFER_GB`;
- `DPP_MAX_DISTILLATION_FRAMES`;
- `DPP_MAX_DISTILLATION_CHUNKS`;
- `DPP_MAX_PAIRS`;
- `DPP_VAL_PAIR_COUNT`;
- `DPP_VAL_EVERY_EPOCHS`;
- `DPP_SKIP_AUTH_EVAL`;
- CPU advisory flags.

Any DP1 dispatch must record these env vars in provenance so a later score cannot be explained by an implicit cap, default cache, or silent device choice.

## Roadmap And Complete Task List

### T0 - Local no-spend closure

1. Re-run DP1 trainer help and smoke after this wiring pass.
2. Run shell syntax on remote driver and recipe wrapper.
3. Regenerate or append readiness summary so the stale `_full_main NotImplementedError` statement is removed from operator-facing surfaces.
4. Add a DP1 command manifest with exact local smoke command, exact full tiny-run command, and exact Modal command template.
5. Add or update tests that prove the remote driver forwards F3/cache/streaming/cap flags.
6. Confirm no hardcoded device-name output path can produce a phantom contest-CUDA score for DP1.

### T1 - First real tiny training run

1. Choose explicit source mode: local chunks, explicit cache, or streamer.
2. Run a tiny full training probe with small caps, for example `DPP_RUN_FULL=1`, `DPP_MAX_DISTILLATION_CHUNKS=1`, `DPP_MAX_DISTILLATION_FRAMES=512`, `DPP_MAX_PAIRS=32`, `DPP_VAL_PAIR_COUNT=4`, and `DPP_SKIP_AUTH_EVAL=1`.
3. Verify `best.pt`, `archive.zip`, `0.bin`, `submission/inflate.sh`, `submission/inflate.py`, `manifest.json`, and `provenance.json`.
4. Run parser and trained-archive no-op consumption checks against the produced archive. The code-level fixture now proves codebook consumption, but every promoted trained archive still needs its own codebook/renderer/residual mutation proof.
5. Build a result-review packet even if auth eval is skipped, classifying the result as training/runtime custody only.

### T2 - Timing smoke on provider

1. Claim lane with `tools/claim_lane_dispatch.py`.
2. Dispatch Modal T4 or A100 timing smoke with the same explicit dataset source and provenance flags.
3. Harvest via canonical recovery path.
4. Record seconds/epoch, max RSS/VRAM, wall-clock, archive bytes, runtime SHA, and all env/config fields.
5. Do not rank or promote from this run unless exact eval artifacts and result-review packet are present.

### T3 - First exact DP1 score

1. Run full enough to produce a byte-closed archive.
2. Run paired auth eval on the same archive/runtime:
   - contest-CUDA;
   - contest-CPU if CPU axis is being used.
3. Run formula recomputation and component review.
4. Build `tac_result_review_packet_v1`.
5. Append evidence row only after engineering forensic audit is clean.
6. If the score is worse than baseline but an audit blocker exists, classify as `indeterminate_engineering_or_config_blocker`, not dead.

### T4 - Composition cells

1. DP1 x A1: exact baseline, composed archive, paired eval, interaction-term measurement.
2. DP1 x YUCR: same, with scorer-blindspot / cost-map analysis.
3. DP1 x PR101/PR106: only after public-runtime reproduction and exact-source custody.
4. Measure whether DP1 stacks with packetIR/compiler/CMA-ES/Optuna/water-filling by comparing deltas under a single archive/runtime pipeline.
5. Promote only if the composed result beats the current operator threshold and survives paired-axis audit.

### T5 - Production-grade DP1

1. Dynamic Comma2k19 chunking modes: frame range, motion class, entropy, saliency, byte size, temporal window.
2. Streamer benchmark and chosen fastest mechanism documented.
3. VQ/VQ-VAE or wavelet codebook alternative if PCA bytes become rate-limited.
4. STC or arithmetic coding for per-pair residual if residual bytes become binding.
5. Federated aggregation with reserved differential-privacy noise parameter and provenance.
6. Public OSS runbook with deterministic data, training, archive, inflate, eval, and exact command reproduction.
7. Submission packet links to repo commit, public repo URL, artifact hashes, configs, seeds, training curriculum, and eval commands.

## Stop / Continue Criteria

Continue DP1 if any of these are true:

- Tier-C density remains across-class and no engineering blocker has invalidated the archive parser;
- tiny real training produces a byte-closed archive and no-op consumption proof;
- DP1 x A1 or DP1 x YUCR has negative interaction term or measurable score improvement;
- production reuse value remains high even if contest delta is small.

Stop only the measured config, not the family, if:

- exact paired CPU/CUDA score regresses against matched baseline;
- the result-review packet has no engineering/config blockers;
- archive/runtime closure, device axis, formula recomputation, and terminal claim all pass;
- reactivation criteria name the next materially different config.

Family-level kill is not currently justified.

## Implementation Hardening Addendum - 2026-05-15T20:05Z

Read-only DP1 audit confirmed the same honest status: DP1 can smoke, parse,
probe, and package proxy artifacts, but it has not produced a real
Comma2k19/Comma10k/BDD100K/Waymo-trained prior, deployment-grade packet, or
legitimate contest CPU/CUDA score.

Code hardening landed after the audit:

1. `src/tac/substrates/pretrained_driving_prior/dataset_source.py`
   introduces a typed `dp1_dataset_source_manifest.v1` contract. It records
   source mode, chunk IDs, SHA-256 coverage, local `video.hevc` hashes, license
   tags, distillation mode, seed, frame caps, and explicit reproducibility
   blockers.
2. `experiments/train_substrate_pretrained_driving_prior.py` now fails closed
   on ambiguous real-source selection. Real `comma2k19` runs must choose
   exactly one source mode: prebuilt codebook, explicit local chunks, local
   cache, or streaming log. `bdd100k` remains rejected in the trainer because
   it is not actually wired.
3. Full DP1 provenance and archive metadata now include
   `dataset_source_manifest`, so a later archive cannot silently detach from
   the pretraining source used to build the codebook.
4. `log_incremental_feeder.codebook_pca_quality_metric` is now directionally
   consistent with plateau logic: it is a deficiency metric where lower is
   better, and `marginal_improvement = previous - current` is positive when
   the prior improves.
5. Cache-mode schedule logs now include final-step chunk paths and SHA-256s;
   streaming-mode schedule logs include the configured per-chunk SHA manifest
   when available.

Verification:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  src/tac/substrates/pretrained_driving_prior/tests \
  src/tac/tests/test_dp1_remote_driver_contract.py \
  src/tac/tests/test_dp1_inflate_consumes_codebook.py \
  src/tac/tests/test_check_209_dp1_contest_video_leakage_caller_check.py \
  src/tac/tests/test_check_210_211_dp1_hardening.py \
  src/tac/tests/test_check_213_comma2k19_canonical_download.py \
  src/tac/tests/test_driving_prior_readiness.py
# 281 passed in 66.37s
```

Current DP1 status after this patch:

- `synthetic_test`: structural smoke/proxy only.
- `comma2k19`: code path is now source-custody hardened, but no real trained
  prior has been produced in this addendum.
- `comma10k`: not a wired DP1 video-pretraining source; it is a segmentation
  image dataset family and would need a separate source adapter/contract.
- `bdd100k`: declared as future optional source, rejected by trainer until a
  real adapter and license gate exist.
- `waymo_open_dataset`: planning-only in readiness manifest; no trainer path.

Next score-bearing DP1 action remains a no-claim, one-source real tiny run:

```bash
PYTHONPATH=src .venv/bin/python experiments/train_substrate_pretrained_driving_prior.py \
  --device cpu --full-cpu --advisory-cpu-explicitly-waived \
  --dataset-name comma2k19 \
  --epochs 1 --batch-size 1 --max-pairs 4 --val-pair-count 1 \
  --max-distillation-frames 128 --max-distillation-chunks 1 \
  --skip-auth-eval \
  --output-dir experiments/results/dp1_comma2k19_onechunk_cpu_advisory_<UTC>
```

This is still not a score claim. It is the first reproducibility-gated step
toward a real pretrained prior artifact.
