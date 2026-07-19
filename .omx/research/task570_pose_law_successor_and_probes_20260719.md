# Task #570 — pose-law successor + 3 exact $0 probes (consume #564 surprises)

Date: 2026-07-19 UTC · Lane: `lane_task570_pose_law_successor_and_probes_20260719`
Status: `research_only=true`; source-derived + `[macOS-CPU/Darwin-arm64 advisory]` measurements; NO launch, NO paid dispatch.
Authority: no contest score / promotion / pointer authority. Pointer `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**.
Source: `.omx/research/v10_frozen_space_surprises_20260719_codex.md` (#564) + `SPEC_v10_integer_plane_vehicle_20260719.md`.

## Verdict table

| # | deliverable | landed / measured | value |
|---|---|---|---|
| 1 | POSE-LAW SUCCESSOR (`pose_global_norm_pooled_dual_v1`) + recomposition probe | LANDED (append-only) + MEASURED | `S_pose(e)=‖e‖/√360` at N=600, `1/√360=0.05270462766947299`; pooled dual admits drop-2 (D=7.53e-5, 2 viol) + drop-3 (D=1.42e-4, 4 viol) that the per-pair veto REJECTS; drop-3 S=457.547 < veto-passed drop-1 S=707.575 by **250.02800483805515** (both rate-dead → gate-logic witness) |
| 2 | U/V RELABEL (test + memo correction + docstring fix) | LANDED | ℓ·u=−0.18790406320541758, ℓ·v=−0.10553922967189727; span{u,v}=(1,1,1)^⊥ ≠ ker(ℓ); principal angle **30.27914784°**, projector distance **0.504213367**; ℓ·δ=0 primal basis verified |
| 3 | GF(257) BM CODER PROBE | MEASURED — formulation KILLED | every one of 64 (role,channel) streams has shortest-LFSR length **L=300 = N/2** (null scale); raw packet **43,280 B ≥ 20,518 B** (loses 2.11×) → verdict_scope: **formulation** (standalone raw-section BM), NOT the BM family |
| 4 | B16/B8 GEOMETRY PROBE | MEASURED `[Darwin-arm64 CPU advisory]` | Seg argmax **0 / 1,572,864** mismatch (d_seg batch-closed); BUT Seg logits max\|Δ\|=**6.771e-05** + Pose first-6 max\|Δ\|=**3.815e-06** ≠ 0 → **EXACT float batch-closure REFUTED**; production cache must preserve literal 37×B16+B8. CUDA axis UNMEASURED (no launch) |

## 1. Pose-law successor + pooled-vs-per-pair recomposition

**DERIVED (source):** `upstream/modules.py:82-84` gives per-pair MSE `q_i = ‖e_i‖²/6`;
`upstream/evaluate.py:81-92` pools to `D_pose = (1/N)Σq_i` and scores `S_pose = √(10·D_pose)`.
At N=600 this is exactly the native error norm `S_pose(e) = ‖e‖₂/√360`, gradient constant-norm
`1/√360 = 0.05270462766947299` away from `e=0` (the `1/√D` D-coordinate blow-up cancels the
`O(‖e‖)` MSE gradient). Therefore the `2.5e-4` "binding crossover" (`5/√(10D)=100 ⇔ D=2.5e-4`)
is a **coordinate-derivative identity**, not a feasibility wall. The evaluator has ONE global
pooled L2 ball `Σ‖e_i‖² ≤ 6N·τ`, NOT 600 per-pair caps — the C4/C9 per-pair veto is a strict
SUBSET (prohibits cross-pair allocation the score allows).

**MEASURED ($0 recomposition, no scorer re-run)** — recomposing the custodied n24 precision-drop
rows (`.omx/research/seg_secant_rd_curve_n24_20260719_v2.json`) under the two feasibility rules,
scoring with the n600-scaled conditional range-payload (`brotli_q11 B/pair × 600 / 37,545,489`):

| point | global D_pose | per-pair viol | per-pair veto | pooled dual (D<2.5e-4) | score S (n600 cond.) |
|---|---:|---:|---|---|---:|
| precision_drop1 | 3.868265e-05 | 0 | PASS | FEASIBLE | 707.575004098 |
| precision_drop2 | 7.527531e-05 | 2 | **VETO** | **FEASIBLE** | 524.636100821 |
| precision_drop3 | 1.415470e-04 | 4 | **VETO** | **FEASIBLE** | 457.546999260 |

The pooled dual admits drop-2 and drop-3 (both global D < 2.5e-4) that the per-pair veto rejects;
the veto's lowest admitted objective (drop-1, 707.575) is **250.02800483805515** HIGHER than
drop-3's (457.547). This exactly witnesses the memo §1 claim: the per-pair veto rejects a
strictly-lower-objective allocation. Both are rate-dead (hundreds) → **gate-logic witness only,
NOT viable archives, not a contest score** (the far-generator spatial-stride rows are infeasible
under BOTH rules). Landed as append-only successor `pose_global_norm_pooled_dual_v1` in
`src/tac/canonical_equations/pose_plane_proximity_law_20260719.py` (corollary_v1 UNCHANGED).

## 2. U/V analysis covectors ≠ primal luma-null plane

Exact fp64 linear algebra from `upstream/frame_utils.py:60-62` (fixture:
`src/tac/tests/test_yuv6_analysis_covectors_vs_primal_luma_null_20260719.py`, 5 tests):
`u·(1,1,1)=v·(1,1,1)=0` ⟹ `span{u,v}=(1,1,1)^⊥`, but `ℓ·u=−0.18790406320541758`,
`ℓ·v=−0.10553922967189727` ⟹ `span{u,v} ≠ ker(ℓ)` (principal angle 30.27914784°, projector
spectral distance 0.504213367; `[ℓ,u,v]` rank 3). The active split in
`tools/c2_perclass_stratum_carrier_analysis.py:345-374` (`gl = gr @ LUMA_HAT`) is a valid
Euclidean `span{ℓ}/ker(ℓ)` diagnostic — corrected the docstring (lines 17-26) + appended a dated
correction section to `.omx/research/frozen_scorer_exact_factorization_20260715.md` (history preserved,
Catalog #110/#113 APPEND-ONLY). `0.504213367` is a worst-case unit-sensitivity attribution bound,
NOT a 50.4% byte/score prediction. Does not reopen the exact 2×2 Pose-visibility law.

## 3. GF(257) Berlekamp-Massey shortest-recurrence coder — standalone raw-section KILLED

verdict_scope: formulation — ONLY the standalone raw-section BM packet on the donor int8 code tensor (L=300=N/2 field-random on all 64 streams; 43,280 B ≥ 20,518 B, loses 2.11×). The BM family/theorem and other tensors are NOT killed.

Donor `levelset_witness_ema_BEST.npz` (= `experiments/results/perclass_bitalloc_witness_20260710/mod32cap_ep650_BEST.npz`,
SHA-256 `6dd28a6e29…`); `code` tensor (1200,32) fp32 → int8 at scale `0.012484688311815262`
(clip ±127) reproduces the custodied tensor SHA-256 `29dca0a738…` exactly. Injective map
`uint8 = int8+128 ∈ {0..255} ⊂ GF(257)`; split into 64 `(frame_role∈{0,1}, channel∈0..31)`
streams of length 600 (rows `role::2`, never flattened/alternated). Berlekamp-Massey over GF(257)
per stream:

- shortest-LFSR length `L` per stream: **min=max=mean=median = 300 = N/2** (the maximal-complexity
  null scale — the quantized latent is field-random; no structure BM can exploit).
- raw packet `Σ(10 + 18·L)/8 = 43,280 B` (10-bit L + 9-bit field elements × (L init + L coeff)),
  metadata-free, vs the `20,518 B` framed section → **loses by 2.11×**; prefilter `mean(L)<141.93` FAILS.

Verdict: `>=20,518 B` ⟹ **kills the standalone raw-section BM formulation** (verdict_scope:
**formulation**, per memo §5). No full parent-archive repack run — the raw section alone already
loses by 2.11×, so no framed/repacked variant can win. The BM theorem, the wider BM family, and
other donor tensors are NOT killed. Confirms memo §5's low-confidence prediction (`mean(L)≈N/2=300`).

## 4. B16/B8 official-tail batch-closure — EXACT float closure REFUTED (advisory)

verdict_scope: formulation — the claim "scorer float outputs are bit-identical across B8-standalone vs first-8-of-B16 batch geometry on Darwin-arm64 CPU-torch fp32" is refuted by direct measurement (Seg logits max|Δ|=6.771e-05, Pose first-6 max|Δ|=3.815e-06 ≠ 0). The SCORED Seg quantity (argmax→d_seg) remains batch-closed (0/1,572,864 mismatches); d_pose is not. Not a family/paradigm negative — the cure is structural (preserve the literal 37×B16+B8 geometry in any cache), and the CUDA axis is unmeasured.

Frozen `DistortionNet` (SegNet EfficientNet-B2 + PoseNet FastViT-T12) loaded from
`upstream/models/{segnet,posenet}.safetensors`, CPU-torch native fp32, 1-thread deterministic.
Official final-8 tail = pairs 592..599 of custodied `gt_n600.npz`; run once as a standalone B8
batch and once as the FIRST 8 rows of a B16 call (padded with pairs 0..7). Compared the 8 target
rows:

- Seg argmax mismatched cells: **0 / 1,572,864** → the scored SegNet quantity (argmax → d_seg) is
  batch-shape invariant on this axis.
- Seg logits max\|Δ\| = **6.771087646484375e-05** (≠ 0); Pose first-6 max\|Δ\| = **3.814697265625e-06** (≠ 0).
- EXACT float batch closure: **FALSE**. Because d_pose consumes the raw fp32 pose outputs (which
  drift ~3.8e-6 with batch shape), d_pose is NOT exactly batch-closed → per memo §6, the production
  cache/eval MUST preserve the literal `37×B16 + 1×B8` schedule; a cache measured uniformly at
  B1/B8/B16/B32 does NOT by itself close the official mixed geometry.

Axis: `[Darwin-arm64 CPU advisory]`. The CUDA axis is UNMEASURED (would require a launch — out of
scope). Recorded as the exact batch-size sequence for C0 cache identity / C11 evaluator custody.

## Triality / wire-in

- **equations leg:** `pose_global_norm_pooled_dual_v1` (append-only successor; producer/consumer lists
  point at C4/C9 + this memo). U/V, BM, and B16/B8 are source-derived corrections/negatives, not new laws.
- **DSL leg:** the successor typed surface is `global_pose_norm.v1` (C4 objective = one pooled dual,
  per-pair telemetry diagnostic). No new lever fires here.
- **DAG leg:** C0 pose-aggregation reconciliation (this successor) precedes C4; the GF(257) probe
  branches only into C6 and is closed (formulation-scoped kill); B16/B8 feeds C0 cache / C11 custody.
- Pointer delta: **none**. No sacred-run bytes written.
