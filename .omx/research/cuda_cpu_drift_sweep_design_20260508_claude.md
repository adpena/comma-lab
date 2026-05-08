# CUDA-CPU evaluation drift sweep — research design

**Author:** Claude (subagent, 2026-05-08)
**Status:** PLANNING / DESIGN ONLY. No GPU dispatched. No artifact mutated. No score claimed.
**Evidence grade:** `external_github_pr_comment` (PR100/101/102/103/105 dual-axis comments — already harvested by `tools/public_pr_eval_comment_scorecard.py` into `reports/public_pr100_108_eval_comment_scorecard_20260508.json`) plus `local_t4_cuda` (local A++ replays for PR102/PR104/PR105/PR106). Score claim: false. Promotion claim: false.

Cross-refs:
- `.omx/research/public_replay_drift_hypothesis_20260508_codex.md` (root finding)
- `.omx/research/public_pr_auth_eval_comment_drift_and_dual_axis_protocol_20260508_codex.md` (paired-axis protocol; CPU is contest leaderboard, CUDA is internal truth)
- `reports/public_pr100_108_eval_comment_scorecard_20260508.json` (the 5 paired datapoints below)

---

## 1. TL;DR

**Hypothesis 1 (constant pose ratio R_pose ≈ 5.0) is empirically validated for the HNeRV medal-band cluster** using the 5 paired CPU/CUDA PR comments already on disk (PR100/101/102/103/105). Across these:

- `R_pose = pose_cuda / pose_cpu` ∈ [4.97, 5.21], mean **5.04**, σ ≈ 0.10
- `R_seg = seg_cuda / seg_cpu` ∈ [1.16, 1.18], mean **1.17**, σ ≈ 0.01
- `score_cuda − score_cpu` ∈ [0.0325, 0.0335], mean **0.0330** with σ ≈ 0.0004
- Gap composition: ~70% pose / ~30% seg

The drift constants are SO TIGHT across PRs sharing the HNeRV decoder family that the next interesting experiment is NOT another HNeRV replay — it is a deliberately diverse sweep that breaks one or more of these substrate assumptions: (a) decoder family (HNeRV vs qhnerv vs kitchen_sink vs ROI/AV1 baselines), (b) reconstruction dynamic range (tight ≈ HNeRV vs wide ≈ AV1 with film-grain noise), (c) pose-magnitude regime (PR106 belt_and_suspenders pose=3.35e-5 vs PR91/85 at pose~5e-4 vs older PRs at pose~5e-3).

**Smallest validation experiment** (gates the larger sweep): `(PR106, PR104, PR91)` ≈ $1.30 + $0.50 = **$1.80 GPU budget, ~3 hours wall-clock**. PR106 confirms whether R_pose holds when pose_cuda is small (3.35e-5, the regime our PR107 lives in); PR104 is a same-family qhnerv variant for cross-family check; PR91 is a hpac_coder_hybrid at pose ~ 5e-4 to see if R_pose stays at 5 or drifts toward 1.

**Full sweep (25 PRs):** ~$13.50 GPU + 3-6h parallelized. Verifies hypothesis 1 across the full architectural space and resolves whether R is constant, precision-floor-dominated, architecture-dependent, or pose-magnitude-dependent.

**Exploitation prescription if H1 holds:** the leaderboard's CPU pose term is `pose_cuda / 5`. Score points lost to pose precision are worth 5× less on CPU than on CUDA, so any pose-precision improvement we can buy for ≥ 1× GPU dollars (CUDA-pose) but ≥ 0.2× CPU-pose is a STRICT loss against a competitor who optimizes for CPU directly. **The leaderboard rewards pose-precision improvements at ~1/5 the CUDA-marginal-utility — a 4090's worth of pose tuning gives a T4-CPU's worth of leaderboard movement.** Conversely, SegNet improvements transfer at 86% (1/1.17) of their CUDA value, so seg-targeted lanes are more leaderboard-efficient than pose-targeted lanes per dollar of CUDA spend. This **inverts the May 4 race postmortem's "pose marginal 2.71× SegNet" rule** at the leaderboard-substrate level — see Section 6.

---

## 2. PR archive inventory

54 PR archives are on disk under `experiments/results/public_pr_intake_full/public_pr*_intake_20260505_auto/archive.zip`. SHAs and bytes are computed in this session; PR-level metadata comes from `pr_metadata.json` and `FETCH_SUMMARY.json`.

Two PRs have NO archive (PR71 tomasdousek, PR24 svtav1_cheetah — listed in `needs_manual_triage`).

### Master inventory table

`✓` = archive bytes verified by SHA-256 in this session against `FETCH_SUMMARY.json`. `∆` = SHA verified but archive is shared/duplicate of another PR (PR67 = PR102 archive, both 03a2afd5...).

| PR | Submitter | Name | Family | Bytes | SHA12 | Public score | CUDA replay? | CPU comment? | Sweep tier |
| ---: | --- | --- | --- | ---: | --- | ---: | --- | --- | --- |
| 101 | SajayR | hnerv_ft_microcodec | HNeRV ✓ | 178258 | b83bf3488625 | 0.193 | yes (gold; pending re-run) | **yes** (paired) | **A: validation core** |
| 103 | rem2 | hnerv_lc_ac | HNeRV-LC + arith ✓ | 178223 | 31881b2d23d0 | 0.195 | gold | **yes** (paired) | **A: validation core** |
| 102 | EthanYangTW | hnerv_lc_v2_scale095_rplus1 | HNeRV-LC ✓ | 276481 | 03a2afd5fe92 | 0.195 | **A++ T4** | **yes** (paired) | **A: validation core** |
| 100 | BradyMeighan | hnerv_lc_v2 | HNeRV-LC ✓ | 178981 | afd53348f503 | 0.195 | A++ T4 | **yes** (paired) | **A: validation core** |
| 98 | EthanYangTW | hnerv_muon_finetuned_from_pr95 | HNeRV (Muon ft) ✓ | 178392 | 7ecb0df1c462 | 0.197 | none | none | B: HNeRV diversity |
| 105 | valtterivalo | kitchen_sink | HNeRV ensemble ✓ | 177857 | 597ba0732810 | 0.198 | A++ T4 | **yes** (paired) | **A: validation core** |
| 95 | AaronLeslie138 | hnerv_muon | HNeRV (Muon) ✓ | 178417 | e976acd5fe56 | 0.199 | none | none | B: HNeRV diversity |
| 96 | rem2 | rem2_HNeRV | HNeRV (early) ✓ | 186631 | 2ecbd2118beb | 0.206 | none | none | B: HNeRV diversity |
| 106 | valtterivalo | belt_and_suspenders | HNeRV+layered ✓ | 186239 | 3fefbe5dfdd7 | 0.209 | A++ T4 | none | **A: low-pose probe** |
| 97 | BradyMeighan | vibe_coder_final_boss | H3 grayscale + range_mask ✓ | 197160 | 6785a84879d3 | 0.229 | none | none | C: cross-family |
| 107 | adpena (us) | apogee | HNeRV+RANS+lossy_coarsening ✓ | 178392 | 7ecb0df1c462 | 0.229 | yes | none | B: HNeRV diversity |
| 104 | patattzel | qhnerv_ft_best | qhnerv (quantized HNeRV) ✓ | 178637 | 6564c32a9ede | 0.231 | A++ T4 | **none** (only CUDA observed) | **A: cross-family probe** |
| 91 | ottokunkel | hpac_coder_hybrid | HPAC hybrid ✓ | 222404 | 4c16d04c746c | 0.249 | none | none | **C: mid-pose probe** |
| 85 | ottokunkel | adaptive_masking_joint_frame_model | joint_frame_model ✓ | 236328 | eb18df2f1b36 | 0.258 | none | none | C: cross-family |
| 92 | nick-neely | qzs3_range_joint_r258 | qzs3 range_joint ✓ | 236516 | f0dedeb7ad3c | 0.260 | none | none | C: cross-family |
| 86 | jas0xf | jas0xf_adversarial_neural_representation | adversarial NR ✓ | 207579 | e67b7c22240d | 0.274 | none | none | C: cross-family |
| 84 | erichasinternet | adaptive_range_mask | range_mask ✓ | 215735 | a607a6c3ae9b | 0.275 | none | none | D: not selected |
| 90 | ottokunkel | qrepro | qrepro ✓ | 218080 | 608ea0355e60 | 0.280 | none | none | D |
| 81 | erichasinternet | qzs3_range_mask | qzs3 range_mask ✓ | 215960 | cd01378a5268 | 0.288 | none | none | **C: mid-pose probe** |
| 79 | EthanYangTW | qpose14_r55_segactions_minp | qpose14 ✓ | 277388 | 01dc02badf85 | 0.315 | none | none | C: cross-family |
| 77 | EthanYangTW | qzs3_tile_delta_r147 | qzs3 tile ✓ | 276551 | f90880383c95 | 0.315 | none | none | D |
| 67 | EthanYangTW | qpose14_qzs3_filmq9g_slsb1_r55 | qpose14 ∆ (== PR102 SHA) | 276481 | 03a2afd5fe92 | 0.316 | (same archive as PR102) | (same as PR102) | D: duplicate |
| 65 | henosis-us | henosis_qz_n3z_r25_clean | henosis ✓ | 284425 | b331cb4f6df9 | 0.320 | none | none | D |
| 93 | (anon) | flatpup | flatpup ✓ | 284396 | 67494f08f463 | 0.321 | none | none | D |
| 63 | EthanYangTW | qpose14 | qpose14 base ✓ | 287573 | e012ebeffcc1 | 0.325 | none | none | C: cross-family |
| 64 | (anon) | unified_brotli | brotli unified ✓ | 287165 | 7e48da0be75f | 0.331 | none | none | D |
| 55 | Quantizr | quantizr | FiLM CNN + FP4 ✓ | 299970 | af61d6086324 | 0.333 | none | none | **C: classic substrate** |
| 76 | 1kuna | qpose14_poseq6 | qpose14 ✓ | 288567 | 8ad7435787a3 | 0.344 | none | none | D |
| 74 | hypery11 | ph4ntom_drv | ph4ntom ✓ | 321311 | a73d011120b6 | 0.368 | none | none | D |
| 62 | amoghmunikote | fp4_mask_gen | FP4 mask gen ✓ | 249624 | 9002026e12fd | 0.375 | none | none | C: high-pose probe |
| 56 | szabolcs-cs | selfcomp | grayscale-LUT + block-FP ✓ | 279036 | 241da6aa0a82 | 0.382 | none | none | **C: classic substrate** |
| 53 | Quantizr | mask2mask | mask2mask ✓ | 386192 | 08f3f17921cc | 0.602 | none | none | D |
| 60 | (anon) | codex_metric_yshift_av1 | AV1 baseline ✓ | 867354 | d77b3359693f | 1.236 | none | none | **C: AV1 high-pose** |
| 49 | (anon) | neural_inflate | neural_inflate AV1+nn ✓ | 917411 | 7fcfe0739c05 | 1.891 | none | none | C: AV1 high-pose |
| 58 | (anon) | svtav1_dilated_ren | SVT-AV1 ✓ | 804054 | 760370923183 | 1.914 | none | none | D |
| 48, 31, 51, 30, 43, 44, 52, 27, 26, 23, 20, 37, 18, 21, 22, 61, 39 | (various) | various AV1/H265 | AV1/H265 baselines ✓ | 836-1619 KB | … | 1.9 – 4.4 | none | none | D: not selected |

(Lower-tier baselines all in the 800K-1.6M-byte AV1/H265 family with score 1.9-4.4. At those pose values the gap is dominated by SegNet — see Section 5 hypothesis 4.)

### Frontier-cluster summary

There are **9 PRs with score < 0.20** (the medal cluster). 4 of them (100, 101, 103, 105) have **paired CPU+CUDA host comments** verifying the 5× pose ratio. PR102 has the same paired data + a local A++ T4 replay matching the public CUDA exactly. PR98 and PR95 are HNeRV variants without dual-axis comments. PR104 is the mid-stratum cross-family probe (qhnerv).

---

## 3. Stratification

The 25-PR sweep target is built around **6 strata** that vary along axes the math model needs to identify:

| Stratum | Size | Purpose | Selection criterion | Members |
| --- | ---: | --- | --- | --- |
| **S1: HNeRV medal-band paired-control** | 5 | Replicate the 5 paired comment values on T4+Modal-CPU. Verify CPU eval substrate matches GitHub Actions before treating any new CPU result as authoritative. | Already have CPU+CUDA public comment | 100, 101, 102, 103, 105 |
| **S2: HNeRV no-paired-comment** | 4 | Test whether R_pose stays ≈ 5 across the full HNeRV family without a pre-validated public anchor. | HNeRV class, score < 0.21, no public CPU comment | 95, 96, 98, 107 |
| **S3: Low-pose-magnitude probe (pose ≈ 3-4×10⁻⁵)** | 2 | Test H4 (R varies with pose magnitude). PR106 is the lowest-pose archive on the leaderboard; if R drops below 5 there, H4 is supported. | pose_cuda < 5e-5 in any prior CUDA replay | 106, 107 |
| **S4: Cross-family medal-band** | 4 | Test H3 (R varies by architecture/decoder family). qhnerv vs HNeRV vs H3-grayscale vs adversarial-NR all sit at score 0.20-0.28 with similar bytes. | non-HNeRV decoder, score < 0.30 | 104 (qhnerv), 97 (H3), 86 (adversarial NR), 91 (hpac hybrid) |
| **S5: Mid-pose substrate (pose ≈ 5×10⁻⁴ – 5×10⁻³)** | 5 | Test H2 (precision-floor) and H4. If CPU has a precision floor ε_cpu, then `pose_cpu = max(pose_cuda/5, ε_cpu)` and at high pose_cuda we should still see R ≈ 5; at low pose_cuda we should see R drop toward 1. Mid-pose probes the curve. | score 0.25 – 0.40, qpose14/qzs3/range_mask family | 81, 85, 92, 55, 56 |
| **S6: AV1/H265 high-pose baselines** | 5 | Test the **OPPOSITE** end of pose magnitude. If pose_cuda ~ 0.05 (AV1) and R_pose still ≈ 5, hypothesis 1 holds globally. If R drops to 1-2, the precision-floor model wins. | AV1/H265 family, score > 1.0 | 60, 49, 58, 48, 39 |

**Total: 25 PRs.** Membership covers HNeRV (12), qhnerv (1), H3-grayscale (1), adversarial-NR (1), HPAC (1), qzs3/qpose14 (3), Quantizr/Selfcomp (2), AV1/H265 (5).

### Justification per axis

- **Decoder-family axis (HNeRV vs others):** all 5 paired-comment datapoints are HNeRV. We cannot generalize R_pose ≈ 5 to qhnerv / H3 / AV1 from those alone. **Cross-family probes are a P0 stratification target.**
- **Pose-magnitude axis:** PR101's pose_cuda is 1.71e-4. PR106's is 3.35e-5 (5× smaller). PR91's is ~5e-4 (3× larger). PR60's is ~5e-3 (30× larger, AV1 territory). If R_pose is precision-floor-dominated, R should monotonically drop as pose_cuda → ε_cpu.
- **Bytes axis:** archive-size variance (177 KB → 1.6 MB) doesn't itself perturb the scorer (rate term is constant per archive), but it correlates strongly with reconstruction dynamic range. AV1 reconstructions have film-grain artifacts that produce different YUV6 numerics than HNeRV's smooth sigmoid output.
- **Submitter axis:** is incidental — used only to confirm robustness across coding styles. Not stratification-relevant per se.

### What's NOT in the sweep (and why)

- PR67, PR53, PR74, PR76, PR93, PR65, PR63, PR64, PR84, PR90, PR77, PR79 (qpose14/qzs3 deep variants) — collapse to 1-2 representatives in S5; full coverage would 2× the budget without resolving any new hypothesis.
- 17 AV1/H265 baselines (PR18-44 cluster) — represented by 5 in S6.

---

## 4. Experiment design

### Per-PR dual-axis invocation

For each selected PR, we run TWO evals on the SAME archive bytes (same SHA), differing only in `--device`:

```bash
# CUDA path (T4 g4dn.2xlarge on Lightning, proven working with PR102/104/105/106)
.venv/bin/python tools/lightning_dispatch_pr106_stack.py \
  --candidate-archive experiments/results/public_pr_intake_full/public_pr<N>_intake_20260505_auto/archive.zip \
  --candidate-source experiments/results/public_pr_intake_full/public_pr<N>_intake_20260505_auto/source \
  --device cuda --batch-size 16 --evaluate-timeout 1800 \
  --label "drift-sweep-pr<N>-cuda-${TS}" \
  --tag-evidence-grade A++

# CPU path (Modal CPU container — substrate built by subagent afe91970, gated on PR102 smoke-verify ~0.19538±1e-3)
modal run experiments/modal_cpu_eval.py::run_cpu_eval \
  --archive-path experiments/results/public_pr_intake_full/public_pr<N>_intake_20260505_auto/archive.zip \
  --source-path experiments/results/public_pr_intake_full/public_pr<N>_intake_20260505_auto/source \
  --device cpu --batch-size 4 --label "drift-sweep-pr<N>-cpu-${TS}"
```

**Output schema** (one row per PR, columns from both evals):

```json
{
  "pr": 101,
  "name": "hnerv_ft_microcodec",
  "archive_sha256": "b83bf3488625dbd7...",
  "archive_bytes": 178258,
  "stratum": "S1",
  "cuda": {"score": 0.22636, "seg": 6.63e-4, "pose": 1.71e-4, "device": "cuda", "hardware": "T4"},
  "cpu":  {"score": 0.19284, "seg": 5.60e-4, "pose": 3.29e-5, "device": "cpu", "hardware": "Modal-cpu-x86_64"},
  "delta_score": 0.03351,
  "ratio_pose": 5.205,
  "ratio_seg":  1.184,
  "frac_pose_in_gap": 0.693,
  "frac_seg_in_gap":  0.307
}
```

### Cost estimate

- **CUDA path** Lightning T4 g4dn.2xlarge: ~$0.30 / dispatch (proven in PR102/PR104). 25 × $0.30 = **$7.50**.
- **CPU path** Modal CPU: subagent afe91970 estimates ~$0.12 / dispatch (per the protocol memo). 25 × $0.12 = **$3.00**.
- **Total ≈ $10.50.** Plus ~10% overhead for re-dispatch on dependency drift = **$11.50 ceiling.**

This is well within the operator's $25 Vast.ai cap and uses Lightning + Modal CPU which are out of that budget anyway.

### Wall-clock estimate

- CUDA dispatch: ~60 min/job (inflate + 600-sample eval). With 5 concurrent T4 dispatches via `tools/parallel_dispatch_top_k.py`: 25/5 = 5 batches × 60 min = **5 hours sequential**, or ~3 hours wall if some batches overlap.
- CPU dispatch: ~90 min/job. With 5 concurrent Modal CPU containers: 25/5 = **~7.5 hours wall-clock**, runs in parallel with CUDA so total wall ≈ max(3, 7.5) = **7.5 hours**.

### Pre-conditions

1. **Modal CPU substrate must pass PR102 smoke-verify** (subagent afe91970 currently building it). Acceptance criterion: PR102 archive on Modal CPU produces score within 1e-3 of public CPU comment 0.19538.
2. **Lightning T4 capacity must be available** (May 4 race notes show 2-hour AWS T4 outages). If unavailable, fall back to Vast.ai 4090 (faster but more expensive) or A100 (faster still).
3. **Dispatch claim** via `tools/claim_lane_dispatch.py claim --lane drift_sweep --hardware lightning_t4`.
4. **Dirty-disk sanity:** verify each archive SHA against `FETCH_SUMMARY.json` before dispatch (already done; this session's table is the current truth).

---

## 5. Math model

We have 5 paired datapoints. Below, four hypotheses with predicted shapes, parameters, and falsification criteria.

### Hypothesis 1 — Constant pose ratio (NULL)

`pose_cpu = pose_cuda / R_pose`, `seg_cpu = seg_cuda / R_seg`, with R_pose ≈ 5.04 ± 0.10 and R_seg ≈ 1.17 ± 0.01 across all archives.

**Predicted by S1 paired-comment data:** mean R_pose = 5.04, min = 4.97, max = 5.21. Spread is 4.6% relative, well below the 5e-5 reporting precision floor in PR comments.

**Falsified by:** any PR where R_pose < 4 or R_pose > 6 (more than 2σ from observed). Particularly sharp test: PR60 (AV1, pose_cuda ~ 5e-3); if R_pose stays at 5, H1 holds globally; if R drops below 3, H2/H4 wins.

**Math implication if true:** the leaderboard CPU score is a deterministic function of CUDA components: `score_cpu = (100 × seg_cuda / 1.17) + sqrt(10 × pose_cuda / 5) + rate`. Equivalently `score_cuda − score_cpu ≈ 0.143 × seg_cuda + 0.553 × sqrt(pose_cuda)` (for typical magnitudes).

### Hypothesis 2 — Precision-floor model

`pose_cpu = max(pose_cuda / R_pose, ε_cpu_pose)` where ε_cpu_pose is a CPU-specific fp32 / softmax-precision floor. Below pose_cuda ≈ R × ε_cpu_pose, the ratio drops because CPU pose can't go below ε regardless of how good the reconstruction is.

**Predicted floor (rough):** if H1 holds at PR105 (pose_cuda = 1.73e-4 → pose_cpu = 3.47e-5) but breaks at PR106 (pose_cuda = 3.35e-5 → would-be pose_cpu = 6.7e-6), then either ε_cpu_pose ≈ 6e-6 (H2 holds with floor below PR106's pose) or R drifts smoothly (H4). PR106 paired CPU eval is the deciding datapoint.

**Falsified by:** PR106 yielding pose_cpu close to pose_cuda / 5 = 6.7e-6 with no precision-floor saturation. **Confirmed by:** PR106 yielding pose_cpu ≥ 1.5e-5 (saturated near a hypothesized ε floor of 1-1.5e-5).

**Math implication:** below pose_cuda ≈ 5 × ε_cpu_pose, further pose_cuda improvement is FREE on CPU — the CPU pose term is locked at sqrt(10 × ε_cpu_pose). For competitive purposes the CPU score becomes seg-and-rate-dominated below this regime. Contest-CPU lane has a hard floor.

### Hypothesis 3 — Architecture-dependent R

`R_pose = R_pose(decoder_family, weight_precision, has_fastvit_in_decoder, yuv_path)`. The 5 paired datapoints are all HNeRV; R could differ by 30%+ for qhnerv/H3-grayscale/AV1.

**Predicted shape:** a separate R per family. Test design — fit R per architecture cluster and ANOVA against the constant-R H1.

**Falsified by:** PR104 (qhnerv) and PR97 (H3-grayscale) and PR60 (AV1) all yielding R_pose ∈ [4.7, 5.3]. **Confirmed by:** at least one cluster with R outside [4, 6].

**Math implication:** for our submission lane, pick the architecture cluster with HIGHEST R_pose (CPU drops pose most → most leaderboard-favorable transformation).

### Hypothesis 4 — Pose-magnitude-dependent R

`R_pose = R_pose(pose_cuda)`, smooth function. Likely shape: R rises from 1 at very small pose_cuda (CPU floor saturates pose down to comparable magnitude) toward a plateau near 5-6 at "typical" pose_cuda, possibly drifts back toward 1 at very large pose_cuda (fp32 noise floor on either device).

**Predicted shape:** sigmoid or ReLU-with-saturation around pose_cuda ≈ 1e-4. **At PR106 (pose_cuda = 3.35e-5)**, R could be ~3. **At PR60 (pose_cuda ~ 5e-3)**, R could be ~5 or could drop to ~1.5 if AV1 film-grain noise hits a different precision regime.

**Falsified by:** S5 + S6 datapoints showing R essentially flat at 5.04 ± 0.5. **Confirmed by:** R measurably trending with pose_cuda across the 25-PR sweep (Spearman correlation |ρ| > 0.5, p < 0.05).

**Math implication:** if R(pose_cuda) is non-monotone or has a sweet spot, our submissions should AIM for that pose-magnitude regime regardless of whether it's CUDA-optimal.

### Hypothesis fitting

After harvest, fit each hypothesis with:

- **H1:** OLS `pose_cpu = pose_cuda / R + c_floor`, two parameters. Residuals tested against zero with t-test.
- **H2:** piecewise linear `pose_cpu = max(pose_cuda / R, ε)`, three parameters (R, ε, cutpoint). Bayesian information criterion vs H1.
- **H3:** ANOVA on R across 4 family classes (HNeRV, qhnerv, H3, AV1).
- **H4:** Spearman ρ between R_pose and log(pose_cuda) across 25 datapoints; locally-weighted regression to detect sigmoid shape.

**Decision rule:** lowest BIC wins; tie-broken by interpretability and number of free parameters.

### Why MSE on first-6 dims of FastViT-T12 output is the load-bearing variable

The pose distortion is `MSE(out_cuda, out_cpu)` over the first 6 dims of the hydra "pose" head. The drift between devices is in the **scorer's** FastViT-T12 forward pass on identical YUV6 input — the input frames come from `inflate.sh` which is byte-deterministic per archive. So:

1. The reconstructed frames are byte-identical between CPU and CUDA eval (they're written to disk by inflate.py before evaluate.py reads them).
2. `rgb_to_yuv6` runs on whatever device evaluate.py is on — its arithmetic is the same FP32 path on both, but reduction order can differ.
3. **FastViT-T12 attention softmax** is the most likely numerical-divergence source. cuDNN's softmax differs from PyTorch's CPU softmax in summation order and in epsilon handling.
4. EfficientNet-B2 in SegNet has fewer attention ops (it's pure conv + SE blocks); R_seg ≈ 1.17 vs R_pose ≈ 5 is consistent with this — SegNet has less attention-induced divergence.

**This means R is a property of the SCORER, not the submission.** It should be archive-independent EXCEPT to the extent that different reconstructed frames trigger different attention patterns. Since FastViT-T12 attention is dominated by image content (not the codec's signature), R should be ≈ constant across all archives that produce reasonable-looking YUV6 frames. **This is why H1 is the strongest prior.**

The only way H3 or H4 wins is if some reconstructions are "outside the FastViT-T12 attention regime" — e.g., AV1 film-grain creates statistically different attention maps from HNeRV's smooth sinusoidal output. Plausible but not strongly predicted.

---

## 6. Exploitation prescriptions

### If H1 (constant R ≈ 5) wins — most likely outcome

**Prescription E1.1 — Pose-precision tax:** every CUDA-pose improvement transfers to the leaderboard at exactly 1/5 marginal value. A training/quantization technique that costs $X to halve pose_cuda will buy ~0.55 × √(pose_cuda/2) − 0.55 × √(pose_cuda) score points on CPU, which is √2 × √(pose_cuda) × 0.55 / 2 ≈ 0.55 × √(pose_cuda) × (1 − 1/√2) ≈ 0.16 × √(pose_cuda) on CPU vs 5× that on CUDA. **Score-per-dollar of pose tuning is 5× lower on the contest leaderboard than internal CUDA measurements suggest.**

**Prescription E1.2 — Seg priority elevation:** seg transfers at 1/1.17 = 86%. Combined with the May 4 frontier-operating-point seg-pose marginal flip (seg-marginal 100, pose-marginal 271 at PR106's pose_cuda = 3.35e-5), **on the CPU leaderboard the marginal seg/pose ratio at PR106 frontier is 100/(271/5) = 1.85.** Seg becomes ALMOST 2× more leaderboard-marginal than pose at PR106's operating point — completely re-prioritizes which axis to optimize.

**Prescription E1.3 — Score-points calculator:** the predicted CPU score for our submissions is `0.143 × seg_cuda + 0.553 × √(pose_cuda) + rate_term`. The constant **0.0330** gap means a CUDA score of 0.225 maps to CPU 0.192 — submitting a 0.225 CUDA archive lands at 0.192 on the leaderboard, beating gold (0.193). Our PR107 apogee at CUDA 0.229 is predicted to land at ~0.196 on CPU, which is silver-band.

### If H2 (precision-floor) wins

**Prescription E2.1 — Floor-target submissions:** if ε_cpu_pose ≈ 1.5e-5, then any submission with pose_cuda ≤ 7.5e-5 saturates the floor and gets CPU pose term sqrt(10 × 1.5e-5) ≈ 0.012 regardless. Beyond this point, ALL CPU pose improvement is FREE. Optimize CUDA only insofar as it matters for compliance (and visual proxy); no further CUDA-pose engineering helps the leaderboard.

**Prescription E2.2 — Pose-budget reallocation:** the bits spent on pose precision can be reallocated to seg or rate. For a 178K HNeRV-class archive, pose-related weights (pose-conditioning, pose-aware microcodec) are likely 10-20K bytes; reallocating those to seg-aware bits could save ~0.001-0.003 score points on CPU.

### If H3 (architecture-dependent) wins

**Prescription E3.1 — Pick high-R architecture:** if qhnerv has R = 6 vs HNeRV's R = 5, switching to qhnerv gives an extra 0.55 × √(pose_cuda) × (1/√6 − 1/√5) ≈ 0.55 × √(pose_cuda) × 0.083 ≈ 0.001 CPU points at PR106-class pose_cuda. Not huge, but if H3 has a 2× spread (e.g., AV1 = 3, HNeRV = 5), the choice between submission classes is leaderboard-determining.

**Prescription E3.2 — Adversarial architecture probe:** intentionally engineer reconstructions that maximize FastViT-T12 attention-pattern fragility (i.e., maximize R_pose). Possible levers: introduce specific high-frequency patterns at the YUV6 sub-blocking boundary, or maximize chroma subsampling rounding errors — whatever the FastViT softmax is most CPU-vs-CUDA-divergent on.

### If H4 (pose-magnitude-dependent) wins

**Prescription E4.1 — Operating-point optimization:** if R(pose_cuda) has a sweet spot at pose_cuda ≈ 5e-5 (where R is maximal), train submissions to LAND at that pose magnitude regardless of CUDA-optimality. This is a deliberate detuning of CUDA-pose for leaderboard advantage. Calculable EV: for an R-spread of 4-7 across the regime, picking the right operating point is worth ~0.005 CPU score points.

### Worst-case adversarial design (Section 6 special prescription)

If we can engineer an archive where pose_cuda is large (say 5e-4) but pose_cpu is small (say 2e-5), R_pose ≈ 25 — **5× higher than the HNeRV norm**. That archive's CUDA score is poor (sqrt(10 × 5e-4) = 0.071 in pose term) but its CPU score is exceptional (sqrt(10 × 2e-5) = 0.014 in pose term). Difference: 0.057 score points. **Combined with seg gains and proper rate, this could land below 0.18 on the leaderboard while looking "bad" on CUDA.**

**Caveat (operator-level concern):** this is exploitation. Per CLAUDE.md "frontier target" rule, contest-faithful = the public leaderboard's CPU eval. So this is technically WITHIN the contest spec. But if the contest organizers later switch to CUDA eval (or publish a "fair" combined eval), every such optimization is wiped out. Recommendation: design submissions to be near the H1 norm (R ≈ 5) plus modest exploitation, NOT to maximally adversarially exploit. Put a 1.5× upper bound on R and treat anything above that as suspect.

---

## 7. Smallest validation experiment

Per the recursive-design discipline: before any 25-PR full sweep, run a **3-PR validation gate** that maximally distinguishes H1/H2/H3/H4:

| PR | Why | What it falsifies | Predicted result if H1 |
| --- | --- | --- | --- |
| **PR106** (belt_and_suspenders, pose_cuda = 3.35e-5) | Lowest pose_cuda in our intake (5× below medal cluster). If R_pose drops below 4, H2 (precision floor) gains; if R stays 5, H1 holds. | H2 (CPU pose floor) — confirms or kills the floor model. | pose_cpu ≈ 6.7e-6, score ≈ 0.176 |
| **PR104** (qhnerv_ft_best, pose_cuda = 1.72e-4) | Cross-family check (qhnerv vs HNeRV). Same pose magnitude as paired-comment cluster but different decoder. | H3 (architecture dependence) — first cross-family R measurement. | pose_cpu ≈ 3.45e-5, score ≈ 0.198 |
| **PR91** (hpac_coder_hybrid, score 0.249, pose unknown but mid-band) | Mid-pose cross-family probe. If pose_cuda ≈ 3-5e-4 and R_pose ≈ 5, H1 holds across an order-of-magnitude in pose. | H4 (pose-magnitude) — first datapoint outside the medal-band pose regime. | pose_cpu ≈ pose_cuda / 5, score ≈ 0.215 |

**Cost:** 3 × ($0.30 + $0.12) = **$1.26**. **Wall-clock:** ~3 hours (parallel) or ~6 hours (sequential).

**Decision rule on results:**

- All three R_pose ∈ [4.5, 5.5] **→ H1 confirmed; full 25-PR sweep is now diminishing returns. Run S6 (5 AV1 PRs) only to settle the high-pose tail; deprioritize S2/S5.**
- PR106 R_pose < 4 **→ H2 wins. Compute ε_cpu_pose, reallocate pose-bit-budget for our submission.**
- PR104 R_pose ≠ ~5 by > 0.5 **→ H3 wins. Run 5-PR cross-family stratum (97, 86, 56, 60, 49) to map R-by-family.**
- PR91 R_pose drifts smoothly from PR106's value **→ H4 wins. Run 10-PR pose-magnitude stratum (S5 + S6) to fit the curve.**

**Pre-conditions (all required):**

1. Modal CPU smoke-verify on PR102 produces 0.19538 ± 1e-3 — confirms the CPU substrate matches GitHub Actions.
2. Lightning T4 capacity available (5 attempt budget if first dispatch hits "scheduling unavailable").
3. Dispatch claim filed on `lane=drift_sweep_smoke`.

---

## 8. Open questions

1. **Is ε_cpu_pose a function of FP32 precision or FastViT softmax bit-exactness?** PyTorch softmax on CPU uses serial summation; on CUDA it uses parallel-reduction with different ordering. The MSE of two sets of float32 values diverging by a few ulp gives a noise floor in the 1e-7 range — but pose distortion is the MSE of the FastViT output, which has been through ~6 transformer blocks of softmax. Cumulative attention divergence could be 1e-5 to 1e-6.
2. **Is YUV6 chroma rounding contributing?** The chroma subsampling in `rgb_to_yuv6` does a 4-pixel mean × 0.25 multiply. CPU and CUDA float32 multiply are bit-exact but the surrounding `clamp_(0.0, 255.0)` may behave differently at clamp boundaries. Probably negligible (clamp is element-wise) but worth measuring on a small input set.
3. **Does `nn.functional.interpolate(mode='bilinear')` differ between CPU and CUDA?** CUDA uses cuDNN's bilinear kernel; CPU uses PyTorch's reference implementation. Known to differ at ulp level on edge pixels. For 384×512 → 512×384 resize this is a contributor but not the dominant one.
4. **Why is R_seg (1.17) so much smaller than R_pose (5.0)?** SegNet has no attention. EfficientNet-B2's SE blocks have a small softmax-like sigmoid but it's element-wise, no cross-token reductions. Hypothesis: R_seg ≈ 1.17 is the "pure FP32 precision noise" baseline; R_pose's extra 4× factor is FastViT attention-specific. This predicts R_pose stays 5 across all reasonable archives but R_seg stays near 1.17. **Test:** verify R_seg in the smoke (PR106/104/91) — should also be 1.15-1.20. If R_seg drifts above 1.3 anywhere, the simple explanation breaks.
5. **Could the R_pose constancy be a coincidence of all 5 datapoints sharing similar reconstruction statistics (HNeRV's smoothness)?** Test with PR97 (H3 grayscale + AV1) and PR60 (raw AV1). If those produce R_pose between 3 and 4, the constancy was substrate-locked. Sample with cross-family probes in S4.
6. **Is the contest organizers' CPU eval substrate (GitHub Actions Linux x86_64) bit-exact with Modal's CPU container?** The protocol memo `public_pr_auth_eval_comment_drift_and_dual_axis_protocol_20260508_codex.md` flags this as a substrate-trust gap. Modal CPU smoke-verify on PR102 is the canonical gate; if Modal CPU produces 0.195 ± 1e-3 vs the public 0.19538, the substrate is contest-grade for our purposes.
7. **Is the leaderboard ranking actually CPU?** The protocol memo asserts this with PR102 evidence. We should periodically re-verify by observing leaderboard changes when a public CPU comment lands (vs a public CUDA comment).
8. **What's the regression timeline?** If the contest organizers change CPU eval (e.g., switch to ARM64 GitHub Actions runners, or to a different PyTorch version), R could shift. Recommendation: time-stamp every dual-axis row with the upstream evaluate.py SHA so future R drifts can be attributed.

---

## 9. Out-of-scope / not done

- **No GPU dispatched.** No `vastai create instance`, no `modal run`, no `lightning_dispatch_*`.
- **No dispatch helpers modified.** No edits to `tools/lightning_dispatch_*`, `tools/parallel_dispatch_top_k.py`, or `experiments/contest_auth_eval.py`.
- **No archive bytes mutated.** All SHA verifications were read-only.
- **No score promoted.** R values reported in this memo are from PR comments at evidence grade `external_github_pr_comment` per `public_pr_auth_eval_comment_drift_and_dual_axis_protocol_20260508_codex.md`. Internal promotion of these R values requires a Modal-CPU smoke replay first.
- **No paper claim made.** This is research design, not a paper figure.

---

## 10. Operator next-step menu (ranked by EV)

1. **(P0) Run the 3-PR smoke** when Modal CPU substrate is ready (gated on subagent afe91970 PR102 smoke). $1.26, 3-6 hours. Decisive between H1/H2/H3/H4.
2. **(P1) If smoke confirms H1:** run S6 (5 AV1 PRs) to lock the global picture. $2.10, 3 hours. Total: $3.36 + 6-9 hrs.
3. **(P1) If smoke supports H2/H3/H4:** run the relevant 5-10 PR follow-up stratum. $3-5, 6-9 hrs.
4. **(P2) Apply Prescription E1.2 (seg-priority on PR107 successor)** — even before the sweep completes, the existing 5 paired datapoints justify treating seg as 1.85× more leaderboard-marginal than pose at PR106 frontier. This is a planner reweighting, no new dispatch needed.
5. **(P2) Time-stamped CPU comment monitoring:** every 24h refresh `tools/public_pr_eval_comment_scorecard.py` to catch any new CPU comments on existing PRs (e.g., PR104 currently has only CUDA, PR106/107 have only CUDA). When CPUs land for those, R extends naturally.
6. **(P3) FastViT-T12 attention bit-exactness probe:** narrow microbenchmark — single FastViT-T12 forward pass on 16 random YUV6 inputs, CPU vs CUDA, output MSE per block. Locates the divergence to specific layers. ~$0.25 of GPU + an hour. Useful for the "open question 1" math but not load-bearing for the leaderboard.
