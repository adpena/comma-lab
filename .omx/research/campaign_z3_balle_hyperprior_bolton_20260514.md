# Campaign Z3: Ballé hyperprior bolt-on (across-class staircase Step 1)

**Operator routing**: zen-floor council Z3 (`feedback_zen_floor_band_v2_post_z1_ablation_20260514.md`) ranked Z3 as the cheapest $2 validation that a recognized class-shift literature anchor (Ballé 2018 scale-hyperprior) reduces A1 bytes without distortion regression. The long-term campaign roadmap (`feedback_long_term_multi_year_campaigns_landed_20260514.md` C5) places this as cooperative-receiver Step 1.

**Tag**: `research_only=true` until empirical smoke + full anchors land. NO score claim. NO promotion.

## 1. lane_id + dispatch-claim plan

- `lane_id`: `lane_z3_balle_hyperprior_bolton_recover_20260514` (post-crash recovery of `lane_z3_balle_hyperprior_bolton_campaign_20260514`)
- `dispatch_platform`: Modal T4
- `dispatch_claim_workflow`:
  1. `tools/claim_lane_dispatch.py claim --lane-id <lane> --platform modal --instance-job-id <call_id> --status pending_modal_smoke`
  2. Smoke fires via `scripts/operator_authorize_substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch.sh`
  3. Remote driver appends terminal status via `claim_lane_dispatch.py claim --force --status completed_z3_balle_remote_driver`
  4. On rc != 0: terminal status `failed_z3_balle_remote_driver_rc_${rc}`

## 2. Source evidence + score-lowering hypothesis

**Source evidence**:
- Ballé et al. (2018) "Variational image compression with a scale hyperprior" ICLR 2018, arXiv:1802.01436 — empirical 10-20% rate savings on natural images vs factorized prior.
- Ballé et al. (2017) "End-to-end optimized image compression" ICLR 2017, arXiv:1611.01704 — foundational entropy-bottleneck architecture.
- MacKay (2003) § 6.7 — arithmetic coding ideal rate.
- Z1 ablation 2026-05-14 (`feedback_z1_mdl_ablation_landed_20260514.md`): A1's scorer-conditional MDL density = 99.29% — encoder is within-class SATURATED, confirming that across-class moves are needed to break the encoder-saturation barrier.

**Score-lowering hypothesis**: A1's `latent_blob` (15,387 bytes per archive) is arithmetic-coded under a FACTORIZED prior (mean=0, fixed scale). The scale-hyperprior replaces this with a per-pair CONDITIONAL Gaussian prior whose scale σ_p is predicted via a tiny MLP `h_s(w_hat_p)`. The conditional Gaussian is TIGHTER for each pair's actual statistics, lowering the rate-y term. The side-info hyper-latent `w_hat` adds overhead, but per Ballé 2018 amortization principle when |y stream savings| > |w + MLP weights|, total bytes drop.

**Predicted ΔS = −0.005 to −0.010** vs A1 0.1928 [contest-CPU 1to1] `[prediction; first-principles-bound]` — this is OPTIMISTIC. At A1's 99.29% density the realistic delta is more likely −0.001 to −0.003 (Z1 EUREKA finding: Tier C σ=0.01 perturbation matches the A1↔PR106 medal-band gap fingerprint of class saturation).

## 3. Timing-smoke command (measures seconds/epoch + seconds/candidate)

```bash
# Local CPU smoke (free; ~0.5s/epoch on M5 Max):
.venv/bin/python experiments/train_substrate_z3_balle_hyperprior_bolton.py \
    --output-dir experiments/results/z3_smoke_local_$(date +%Y%m%dT%H%M%SZ) \
    --epochs 100 --device cpu --smoke

# Modal T4 timing smoke (~$0.30; validates remote integration):
OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=0.50 \
Z3_BALLE_SMOKE_ONLY=1 \
Z3_BALLE_SMOKE_EPOCHS=100 \
./scripts/operator_authorize_substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch.sh
```

Expected timing (local CPU smoke validated):
- 3 epochs / 1764 params / 32 synthetic latents → ~0.05s total
- 100 epochs / 1000 params / 32 pairs → ~3s total
- Modal T4 1000 epochs / 600 real pairs → ~5-10 min (~$0.05-0.10)

## 4. Full-run command (resumable checkpoints + harvest path)

```bash
# Full Modal T4 dispatch (~$2 p50; gated behind smoke-green):
OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=2.50 \
OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1 \
OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=1 \
./scripts/operator_authorize_substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch.sh
```

Harvest path:
- Within 24h: `tools/harvest_modal_calls.py` reads `experiments/results/lane_substrate_z3_balle_hyperprior_bolton_*_modal/modal_metadata.json` and writes `harvested_artifacts/` beside each.
- The `_full_main` path raises NotImplementedError pending Phase 2 council green-up; v1 smoke is the canonical path for this campaign.

## 5. Live provider rate/cost model

| Provider | GPU | $/hr | Z3 smoke epoch/s | Z3 smoke 100-ep cost | Z3 full 1000-ep cost |
|---|---|---|---|---|---|
| Modal T4 | T4 16GB | $0.59 | ~30 | ~$0.30 | ~$2.00 |
| Modal A10G | A10G 22GB | $1.10 | ~80 | ~$0.20 | ~$1.40 |
| Vast.ai 4090 | RTX 4090 24GB | $0.25 | ~140 | ~$0.05 | ~$0.40 |
| Local CPU (M5 Max) | MPS noise | $0.00 | ~10 | $0.00 (no auth eval) | research-only |

**Selected**: Modal T4 (canonical infrastructure; smoke + full both under $2.50 hard cap).

## 6. Byte-closed archive/export/inflate plan for promotion

**Archive grammar (Z3HP1 sidecar appended to A1 base)**:
```
[A1 wire format]
  + [Z3HP1 sidecar]:
      magic         : "Z3H1"  (4 bytes)
      version       : 1       (1 byte)
      n_pairs       : 600     (uint16 LE)
      hyper_dim     : 8       (uint8)
      latent_dim    : 28      (uint8)
      int8_w_scale  : float32 LE
      quant_step    : float32 LE
      min_sigma     : float32 LE
      max_sigma     : float32 LE
      reserved      : 2 bytes
      hyperprior_weights_blob : brotli(int8 MLP weights)
      w_hat_blob              : brotli(int8 hyper-latents)
      residual_blob           : brotli(int8 latent residuals)
```

**Estimated sizes** (param_count=1764, brotli_ratio=0.6):
- Header: ~27 + 10 (length prefixes) = 37 B
- Weights blob: 1764 × 0.6 ≈ 1058 B
- w_hat blob: 600 × 8 × 0.6 ≈ 2880 B
- Residual blob: 600 × 28 × 0.6 ≈ 10,080 B
- **Total sidecar**: ~14,055 B

**A1 latent_blob baseline**: 15,387 B. **Amortization budget**: 1,332 B for conditional-Gaussian rate-y improvement. Per Ballé 2018 the realistic conditional-vs-factorized rate ratio is 0.85-0.90 (10-15% savings on the rate-y component), and A1's latent_blob already contains ~12 KB of rate-y after factorized AC coding. So expected savings ≈ 0.85 × 12000 = 10200 B → bytes saved = 15387 - 14055 - 10200 = -8868 B (NET LOSS if the Z3 codec is independent of A1).

**HOWEVER** the realistic path is: Z3 REPLACES A1's rate-y encoding entirely. The conditional-Gaussian decode + residual_blob is the new rate-y; A1's existing rate-y bytes are NOT shipped. Net savings = A1's rate-y bytes (~12,000 B) − Z3 sidecar bytes (~14,055 B) = -2,055 B amortization deficit BEFORE the hyperprior wins via conditional tightness.

This is why Ballé 2018 amortization principle is critical: **ship the sidecar ONLY when bytes_saved > overhead**, else fall back to byte-identical-to-A1. The trainer enforces this in `pack_composition_archive`.

**Export contract**:
- `experiments/results/<dispatch_id>/output/archive.zip` (deterministic ZIP via Catalog #19)
- `experiments/results/<dispatch_id>/output/submission_dir/inflate.sh` (3-positional-arg Catalog #146 contract)
- `experiments/results/<dispatch_id>/output/submission_dir/inflate.py` (≤ 100 LOC bolt-on Catalog L4)

**Inflate plan**: split composition archive → if sidecar absent, delegate to A1 inflate (byte-identical fallback) → else decode Z3HP1 sidecar → reconstruct A1 latents via conditional-Gaussian decode → delegate to A1's existing decoder for frame reconstruction.

## 7. Stop/continue thresholds

| Phase | Stop if | Continue if |
|---|---|---|
| Local CPU smoke | rate_bits_total NaN / Inf, archive parse fails, inflate fails | rate_bits_total decreases monotonically over 3 epochs, archive parse OK, inflate parses sidecar |
| Modal T4 smoke (100 ep, $0.30) | rc != 0, auth_eval missing, score > 0.250 (catastrophic regression vs A1 0.1928) | rc=0, score in [0.180, 0.250] band |
| Modal T4 full (1000 ep, $2.00) | rc != 0, auth_eval missing, score > 0.200 (clear regression) | rc=0, score < 0.193 (improvement over A1) |
| Export | sidecar overhead > bytes_saved | sidecar overhead < bytes_saved (Ballé amortization principle) |
| Exact eval | score_claim_valid=false, score_axis != contest_cuda | score_claim_valid=true AND score_axis=contest_cuda AND in band [0.180, 0.195] |

**Promotion threshold**: score on contest-CUDA AND contest-CPU (1:1 GHA Linux x86_64) within [0.180, 0.195] AND archive bytes < A1 bytes by ≥ 1KB AND distortion components within ±5% of A1's.

**Strict-honest exit**: at A1's 99.29% saturation density, the realistic empirical outcome is bytes_saved < overhead (sidecar suppressed; byte-identical-to-A1; verdict DEFERRED-pending-research). This is acceptable and provides empirical evidence for the within-class trap hypothesis (Z1 cathedral autopilot ranker v2 anchor).

## Cross-references

- `feedback_zen_floor_band_v2_post_z1_ablation_20260514.md` — operator decision Z3 ranking
- `feedback_long_term_multi_year_campaigns_landed_20260514.md` C5 — long-term roadmap
- `feedback_z1_mdl_ablation_landed_20260514.md` — A1 99.29% saturation anchor
- `feedback_mdl_density_gate_and_autopilot_ranker_landed_20260514.md` — Catalog #219 + class-shift literature-anchor reward
- `.omx/research/recovery_session_20260514_directive_absolute_no_signal_loss_20260514.md` — predecessor crash recovery protocol
- `.omx/research/recursive_no_signal_loss_protocol_20260514.md` — recursive R1-R4 directive extension
- `src/tac/substrates/z3_balle_hyperprior_bolton/architecture.py` — predecessor's tiny scale-hyperprior MLP
- `src/tac/substrates/z3_balle_hyperprior_bolton/archive.py` — predecessor's Z3HP1 wire format
- `submissions/a1/archive.zip` — frozen base substrate
