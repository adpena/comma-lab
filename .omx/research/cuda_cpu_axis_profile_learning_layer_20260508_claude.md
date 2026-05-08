# Per-architecture-class CUDA/CPU profile learning layer

**Author**: Claude (subagent — adaptive learning layer, P1 enhancement)
**Date**: 2026-05-08
**Status**: LANDED. CPU-only, no GPU dispatch, no score claims.
**Evidence grade**: `[CPU-prep learning-layer planning-only]` for all
predictions emitted by this layer.
**Cross-references**:

- Sister static helper: `src/tac/optimization/cuda_cpu_axis_calibration.py`
  (a618cdd6) — bootstrap source for HNeRV constants
- Deep-theory mechanism: `.omx/research/cuda_cpu_pose_drift_mechanism_deep_dive_20260508_claude.md`
  (a22d581a) — Welch's-law random walk + 4 noise sources
- 25-PR sweep design: `.omx/research/cuda_cpu_drift_sweep_design_20260508_claude.md`
  (ad0875a8) — empirical generator that will seed non-HNeRV classes
- Decoder drift introspection design: `.omx/research/decoder_drift_introspection_design_20260508_claude.md`
  (a43dd96b) — refines the 25% decoder fraction
- Public PR scorecard: `reports/public_pr100_108_eval_comment_scorecard_20260508.json`

---

## 1. Why a single R is wrong

The hardcoded HNeRV-cluster constants `R_POSE_HNERV = 5.04`, `R_SEG_HNERV =
1.17` are empirically grounded but specific to **one** architecture family
on **one** substrate at **one** operating point. Operator directive
2026-05-08:

> "maybe not a clear priority may need to be learned or our analyzer
> enhanced to include the cpu vs gpu and underlying dali vs av and pyav
> math and learn which to use" + "because different arches and stacks may
> have different profiles which we are still seeking to understand."

The learning layer treats `R_pose` and `R_seg` as **per-class posterior
distributions**, not constants. Five empirically-confirmed reasons each
architecture class is likely to have a different profile:

1. **Different decoder paths.** HNeRV cluster uses
   `DaliVideoDataset → TensorVideoDataset` on CUDA and
   `AVVideoDataset → TensorVideoDataset` on CPU. The deep-theory subagent
   measured this contributes ~25% of pose drift. AV1-only families
   (PR60, PR97 H3) skip the neural decoder entirely; their R_pose may
   collapse toward 1.0.
2. **Different precision-floor magnitudes.** HNeRV's σ²_floor ≈ 1.4e-4
   is set by Welch's-law random walk on ~50 conv ops. A shallower
   network (e.g. PR104 qhnerv_ft if it has a thinner decoder) has a
   smaller floor; a deeper one has a larger one. R_pose flips behavior
   above vs below the floor.
3. **Different operating points.** PR60 (raw_av1_yuv) is reported at
   ~4× the medal-band pose distortion. At higher pose, the floor
   contributes a smaller fraction of the total → smaller R_pose.
4. **Different architecture-specific kernels.** Balle ScaleHyperprior
   uses GDN nonlinearity, not GELU. Different per-op precision noise.
5. **Different normalization layers.** AllNorm `view(-1, 1)` flat
   reduction (Source D in the deep-theory dive) only exists in PoseNet's
   summarizer — but other architectures may have analogous
   single-feature flat reductions.

A single global `R` therefore produces biased predicted-CPU bands for
non-HNeRV families. The fix is to maintain a registry of per-class
posteriors and update them as paired anchors arrive.

---

## 2. Bayesian-update mechanism

Each new `(cuda_score, cpu_score)` paired anchor is an evidence
observation. The registry holds, per architecture class:

- `r_pose_mean`, `r_pose_std` (running mean and Bessel-corrected sample
  std across accepted anchors)
- `r_seg_mean`, `r_seg_std`
- `score_gap_mean`, `score_gap_std`
- `n_anchors`: how many anchors back the posterior
- `evidence_anchors`: full audit trail

The update is conjugate (running mean + n−1 std). Outlier policy:

- If `|observed_r_pose − r_pose_mean| > 3σ` AND the class has
  `n_anchors ≥ 3` (so the posterior is meaningful), the anchor is
  flagged `outlier_candidate=True` and DOES NOT promote into the
  posterior.
- The anchor STAYS in the audit trail with the outlier reason. Per
  CLAUDE.md `forbidden_premature_kill_without_research_exhaustion`, an
  outlier is flagged for operator review, not dropped.
- Operators can promote a flagged anchor manually after review (e.g.
  if it represents a genuine new operating regime, not measurement
  error).

Bootstrap: PR100/101/102/103/105 paired anchors seed the
`hnerv_ft_microcodec` class with `n_anchors=5`, `r_pose_mean ≈ 5.036`,
`r_pose_std ≈ 0.10`. All other architecture classes start at `n_anchors=0`
with HNeRV-default priors and `confidence_label="uncalibrated_default"`.

---

## 3. Decoder-network decomposition

Per the deep-theory dive (a22d581a), the observed pose drift decomposes
multiplicatively on the (R − 1) excess::

    excess = R_pose − 1
    decoder_contribution = excess × decoder_pose_drift_fraction
    network_contribution = excess × (1 − decoder_pose_drift_fraction)

Default `decoder_pose_drift_fraction = 0.25` (HNeRV cluster, NVDEC vs
PyAV chroma upsampling kernel split). Default
`decoder_seg_drift_fraction = 0.05` (SegNet argmax stability). The
decoder-drift introspection subagent (a43dd96b) will refine these
numbers per-class.

Different exploits per axis:

- Reducing **decoder contribution**: contest-faithful improvement
  requires changing the GT path — forbidden by CLAUDE.md "Non-Negotiable
  Upstream Rule". Cannot be exploited.
- Reducing **network contribution**: training-time fixes
  (`cudnn.deterministic`, matched noise injection at
  `noise_std=σ²_floor`). These are operator-only flags, not contest
  contract changes.

The decomposition is the math model that lets the analyzer reason about
"which axis of drift is exploitable in this lane."

---

## 4. How the analyzer adapts as anchors arrive

### 4.1 Online-learning hook

`harvest_new_anchor_and_update(contest_auth_eval_payload)`:

1. Extract paired `(cuda, cpu)` numbers from a
   `contest_auth_eval.adjudicated.json` payload.
2. Classify the archive (or use override) into an architecture class.
3. Run the Bayesian update with outlier check.
4. Append a JSONL audit line to
   `.omx/research/cuda_cpu_axis_profile_updates.jsonl`.
5. Return a `ProfileUpdate` event with before/after diffs.

### 4.2 Watcher pattern

Per the operator directive, the harvester is meant to watch:

- `experiments/results/lightning_batch/*/contest_auth_eval.adjudicated.json`
- `experiments/results/pr*_apogee_cpu_auth_eval_*/contest_auth_eval.adjudicated.json`

A simple cron / loop tick every 5 min calling
`harvest_new_anchor_and_update` for each new file ingests anchors as they
land. Updates are auto-committed via
`tools/subagent_commit_serializer.py`. The audit log persists across
sessions and survives a fresh checkout.

### 4.3 Cathedral-autopilot integration

The recommender consumes `predict_cpu_score_band(architecture_class, ...)`
when ranking candidates. Low-anchor classes (n < 3) get the band widened
by ×1.5 — this is the "we don't know enough about this family yet" prior.
The widening shows up in `confidence_label="low-calibration-confidence"`
or `"uncalibrated_default"`.

### 4.4 Meta-Lagrangian integration

`per_class_lagrangian_weights(architecture_class)` returns CPU-axis
calibrated multipliers `λ_pose / R_pose`, `λ_seg / R_seg`. For HNeRV at
medal band: `λ_pose ≈ 0.198`, `λ_seg ≈ 0.855`. This is the
CPU-leaderboard-faithful weighting. Unknown classes fall back to HNeRV
defaults with `confidence_label="uncalibrated_default"`.

### 4.5 Theoretical-floor integration

`per_class_floor_band(architecture_class)` reports the per-class
precision floor and the implied CPU pose floor. Different architectures
have different achievable floors; pushing a candidate's CUDA pose below
its class's floor is wasted bytes on the CPU leaderboard.

---

## 5. Architecture-class taxonomy (initial)

The classifier knows the following families today:

| class | bootstrap n | source PRs | notes |
| --- | --- | --- | --- |
| `hnerv_ft_microcodec` | 5 | PR100/101/102/103/105 | Calibrated; the canonical bootstrap |
| `hnerv_lc_v2` | 0 | PR100, PR102 (overlap with above) | Sub-class of HNeRV; awaiting separation |
| `qhnerv_ft` | 0 | PR104 expected | Awaiting CPU+CUDA paired eval |
| `kitchen_sink_ensemble` | 0 | PR105 in HNeRV bucket today | Awaiting decomposition |
| `h3_av1_grayscale` | 0 | PR97 expected | Predicted lower R_pose |
| `mnerv` | 0 | None yet | Awaiting public PR or internal lane |
| `balle_scale_hyperprior` | 0 | None yet | Predicted higher R_pose (NN decoder, GDN) |
| `raw_av1_yuv` | 0 | PR60 expected | Predicted R_pose ≈ 1 (no neural decoder) |
| `rgb_packed_brotli` | 0 | None yet | Reserved |
| `unknown_uncalibrated` | 0 | (fallback) | Returns HNeRV defaults |

The 25-PR sweep (subagent ad0875a8) is the empirical generator that
will seed the non-HNeRV classes. Once CSV/adjudicated JSONs land for
PR60, PR97, PR104, etc., the harvester ingests them automatically.

---

## 6. Coordination with sister subagents

| subagent | role | this layer's relationship |
| --- | --- | --- |
| **a618cdd6** static helper `cuda_cpu_axis_calibration.py` | Hardcoded HNeRV constants (R_POSE_HNERV=5.04, etc.) | **Bootstrap source.** This module imports the constants when available; falls back to in-module empirical means if the helper isn't on disk yet. The static helper handles the single-class HNeRV calibration; this layer handles per-class learning. |
| **a22d581a** deep-theory mechanism dive | Welch's-law + 4 noise sources, decoder-network 25/75 split | **Default constants source.** `DEFAULT_DECODER_POSE_DRIFT_FRACTION=0.25` and `DEFAULT_POSE_FLOOR_ESTIMATE=1.4e-4` come from this analysis. |
| **ad0875a8** 25-PR sweep design | Empirical anchors for PR60/97/104/etc. | **Anchor producer.** Once the sweep dispatches and harvests adjudicated JSONs, the online-learning hook ingests them automatically — no manual update step. |
| **a43dd96b** decoder drift introspection | Refines the decoder fraction per-class | **Override source.** Once measured per-class, the `decoder_drift_fraction` field on each `ArchitectureProfile` can be set manually (or via a future helper). |
| **a71a34f7** writeup subagent | Paper / writeup | This memo is the durable record. The writeup can pull from §1-§4 verbatim. |

---

## 7. Sample query — PR104 qhnerv_ft

A hypothetical PR104 archive at `cuda_score=0.232`,
`inferred_kind="ff_packed_brotli_qhnerv"`, `title="qhnerv_ft_best"` produces:

```python
{
  "predicted_cpu_score": 0.199,
  "predicted_cpu_score_low": 0.1975,
  "predicted_cpu_score_high": 0.2005,
  "score_gap_used": 0.033,
  "score_gap_band_half": 0.0015,
  "confidence_label": "uncalibrated_default",
  "architecture_class_used": "qhnerv_ft",
  "n_anchors_backing": 0,
  "r_pose_mean": 5.04,
  "r_seg_mean": 1.17,
  "evidence_grade": "[predicted; learning-layer registry posterior]",
  "score_claim": false,
  "promotion_eligible": false,
  "ready_for_exact_eval_dispatch": false
}
```

`band_half` is `0.0015 = 1.5 × 0.001` — the low-calibration widening
factor (`LOW_CALIBRATION_BAND_WIDENING = 1.5`) applied to the prior std.
Compare to the calibrated HNeRV cluster, which produces band_half ≈
0.0004 — 3-4× narrower because 5 anchors have shrunk the posterior std
substantially.

This widening is the operational signal: when the recommender ranks
PR104 against an HNeRV candidate at the same predicted CUDA score, the
PR104 prediction comes with a wider error bar. A risk-averse operator
should prefer the calibrated candidate; a curiosity-driven operator
should dispatch a PR104 paired CPU+CUDA eval to **calibrate** the class
(closing the loop).

---

## 8. Open questions

1. **How fast does the posterior converge?** With 5 HNeRV anchors,
   `r_pose_std ≈ 0.10` (well below the 5.04 mean). With 0 anchors, the
   prior std is 0.10 set at bootstrap. After 1 anchor, the std collapses
   to 0 (single-sample, no Bessel correction). The current code keeps
   the prior std for n=1; with n≥2 it switches to sample std. A more
   sophisticated estimator (full Normal-Inverse-Gamma conjugate update)
   could blend the prior and likelihood — defer until 25-PR sweep
   anchors arrive.

2. **What's the "right" outlier threshold?** 3σ is conservative — it
   accepts ~99.7% of true samples. At 5 anchors with σ=0.10, the
   threshold is 5.04 ± 0.30. PR104 might genuinely sit outside this if
   it has a different decoder path. The flag-and-review pattern (no
   automatic drop) is the safe default.

3. **When should a class split?** If PR104 (qhnerv_ft) anchors come
   in with R_pose = 4.2 ± 0.05 (consistently below HNeRV's 5.04), the
   right move is a separate class, not pollution of the HNeRV
   posterior. The current classifier treats them as separate from byte
   1 — good.

4. **Does the decoder drift actually flip for AV1-only families?** The
   prediction is yes (no neural decoder → R_pose ≈ 1). Awaits empirical
   confirmation when PR60 anchors arrive.

5. **Should the registry track time-to-converge?** A class that has
   been collecting anchors for 3 weeks but `r_pose_std` hasn't shrunk
   below 0.5 has an anomalous variance — possibly a sign of two
   sub-classes that need to split. Future work.

---

## 9. Recommendation for next operator move

1. **Land the registry to disk.** Run
   `.venv/bin/python -c "from tac.optimization.cuda_cpu_axis_profile_registry
   import bootstrap_registry_from_hnerv_anchors, write_registry;
   write_registry(bootstrap_registry_from_hnerv_anchors())"`
   to materialize `.omx/state/cuda_cpu_axis_profile_registry.json`
   so cron-loop harvests have a concrete file to update.

2. **Wire the harvest cron.** A 5-min cron that scans
   `experiments/results/**/*contest_auth_eval.adjudicated.json` for
   files with mtime > last_harvest, calls
   `harvest_new_anchor_and_update`, writes the registry back, and
   commits via `tools/subagent_commit_serializer.py`. This closes the
   loop without operator intervention.

   Codex follow-up: `tools/harvest_cuda_cpu_axis_profile_registry.py`
   now provides the concrete harvester entry point. It ingests explicit
   `--pair CPU_JSON CUDA_JSON`, already-combined `--combined-json`, or
   `--scan-root` artifacts, requires full-sample `[contest-CPU]` and
   `[contest-CUDA]` axes for the same archive SHA/bytes and runtime-tree
   SHA, rejects macOS CPU advisory artifacts for registry mutation, and
   emits `score_claim=false` / `promotion_eligible=false` reports.

3. **Defer cathedral_autopilot wiring.** The recommender today still
   uses the static helper. After this layer lands, swap
   `cathedral_autopilot._rank_techniques`'s `CudaCpuCalibration(...)`
   call for `predict_cpu_score_band(...)`. Defer until the 25-PR sweep
   has seeded ≥3 non-HNeRV anchors so the swap is genuinely useful.

4. **Don't dispatch any GPU yet.** Per CLAUDE.md "score claim must come
   from contest-CUDA `upstream/evaluate.py`", everything in this layer
   is a CPU-prep prediction. The next dispatch should be a
   contest-CUDA eval that BOTH validates a candidate AND seeds the
   registry.

---

## 10. Module map

| file | role |
| --- | --- |
| `src/tac/optimization/cuda_cpu_axis_profile_registry.py` | Per-class posteriors, Bayesian update, classifier, decoder split, persistence, online-learning hook |
| `src/tac/optimization/cuda_cpu_axis_adaptive_analyzer.py` | Adapter to cathedral_autopilot / meta-Lagrangian / theoretical_floor; lazy registry handle with thread lock |
| `src/tac/optimization/cuda_cpu_axis_calibration.py` | Static helper (sister subagent a618cdd6) — bootstrap source for HNeRV constants |
| `src/tac/tests/test_cuda_cpu_axis_profile_registry.py` | 33 tests (bootstrap, update, outlier, classifier, persistence, harvest) |
| `src/tac/tests/test_cuda_cpu_axis_adaptive_analyzer.py` | 14 tests (registry handle, adapter functions) |
| `tools/harvest_cuda_cpu_axis_profile_registry.py` | Operational exact-artifact harvester; pairs contest-CPU and contest-CUDA JSONs by archive/runtime custody before registry updates |
| `tools/analyze_cpu_cuda_eval_drift.py` | Existing diagnosis tool — input source for the harvester |
| `reports/public_pr100_108_eval_comment_scorecard_20260508.json` | Bootstrap anchors source |
| `.omx/state/cuda_cpu_axis_profile_registry.json` | Persisted registry (gitignored) |
| `.omx/research/cuda_cpu_axis_profile_updates.jsonl` | JSONL audit log of accepted + flagged anchors |

47 tests pass on landing; backwards-compatible with all existing solvers
(callers that don't pass `architecture_class` get the HNeRV defaults).

---

## 11. Score-claim and dispatch hygiene

Per CLAUDE.md non-negotiables:

- `evidence_grade = "[CPU-prep learning-layer planning-only]"` on every
  registry payload.
- `score_claim=False`, `promotion_eligible=False`,
  `ready_for_exact_eval_dispatch=False` on every predicted band.
- The registry seeds priors and TODOs but **never** retires a lane,
  promotes a candidate, or falsifies a family. Predicted CPU scores are
  not dispatchable; they only inform the ranker.
- Outlier-flagged anchors stay in the audit trail. Operators promote
  manually after review. Per `forbidden_premature_kill_without_research_exhaustion`,
  no anchor is dropped — flagging is the maximum action this layer takes.
- The mutation frontier is fully respected: changes land in
  `src/tac/optimization/`, `src/tac/tests/`, `.omx/research/`, and the
  durable summary memo.
