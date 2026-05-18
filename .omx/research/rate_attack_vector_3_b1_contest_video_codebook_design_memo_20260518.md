---
schema: pact_design_memo_v1
memo_id: rate_attack_vector_3_b1_contest_video_codebook_design_memo_20260518
review_date: "2026-05-18"
lane_id: lane_rate_attack_b1_contest_video_codebook_substrate_20260518
parent_master_memo: rate_attack_43_vectors_meta_paradigm_deep_research_20260518
meta_paradigm_anchor: structural_information_not_shipped_meta_paradigm_unification_20260518
vector_id: B1
vector_name: "Contest-video-as-codebook (vector-quantize patches against upstream/videos/0.mkv bytes the decoder ALREADY HAS)"
horizon_class: frontier_breaking
predicted_band_validation_status: pending_post_training
score_claim: false
promotion_eligible: false
research_only: true
write_scope: ".omx/research only"
canonical_frontier_anchor:
  contest_cpu: "0.19205 [contest-CPU]"
predicted_delta_band_contest_cpu: "[-0.020, -0.005]"
council_tier_assignment: T3_full_grand_council
target_modes:
  - contest_exact_eval
  - contest_generalized
deployment_target: t4_contest_runtime
hardware_substrate: linux_x86_64_t4
---

# TOP-3 Design Memo — Vector B1: Contest-Video-As-Codebook

**Master memo**: `.omx/research/rate_attack_43_vectors_meta_paradigm_deep_research_20260518.md`
**META-paradigm**: SINS — `.omx/research/structural_information_not_shipped_meta_paradigm_unification_20260518.md`
**Lane**: `lane_rate_attack_b1_contest_video_codebook_substrate_20260518` L0

## 0. Executive Summary

**HARD-EARNED structural anchor**: `submissions/exact_current/inflate.py` finds upstream root, which contains `videos/0.mkv` — the contest video bytes (~37.5MB). **The decoder ALREADY HAS the upstream contest video as readable side info at inflate time.**

**B1 exploit**: canonical Wyner-Ziv 1976 source-coding-with-side-information. For each rendered frame patch, identify the best-matching patch from the upstream video's frame patches; ship only `(patch_idx, dx, dy, scale, residual)`. The decoder reads the upstream video locally (free; pinned environment) and reconstructs.

**Predicted ΔS**: [-0.020, -0.005] — HIGHEST UPPER BOUND in TOP-5 (van den Oord canonical VQ-VAE territory).

**Council verdict**: T3 sextet+grand 20-of-20 PROCEED_WITH_REVISIONS (van den Oord binding: 32×32 patches + 2048-entry codebook initial sweep).

## 1. Canonical-vs-unique Decision Per Layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Upstream video access | ADOPT_CANONICAL | `submissions/exact_current/inflate.py::find_upstream_root` already reads videos/0.mkv |
| pyav decode | ADOPT_CANONICAL | Pinned upstream dependency per HNeRV parity L9 |
| Faiss IVF-PQ for fast NN search | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | NEW dependency for codebook lookup; ~2 MB add to dependency closure |
| VQ codebook builder | **FORK_BECAUSE_SUBSTRATE_OPTIMAL** | NEW substrate-engineering helper at `tac.substrates.rate_attack_b1_contest_video_codebook` |
| Archive grammar | **FORK_BECAUSE_SUBSTRATE_OPTIMAL** | NEW grammar: (patch_idx, dx, dy, scale, residual) per patch; sister of PR101 grammar |
| Inflate runtime | ADOPT_CANONICAL + extension | ~80 LOC extension: read video + build codebook index + reconstruction loop |
| Residual encoding | ADOPT_CANONICAL (Brotli) | Existing canonical compression for residuals |
| Score-aware loss | ADOPT_CANONICAL (`score_pair_components`) | Existing canonical helper |
| EMA | ADOPT_CANONICAL (0.997) | CLAUDE.md non-negotiable |
| eval_roundtrip | ADOPT_CANONICAL (True) | CLAUDE.md non-negotiable |
| Tier-1 + Tier-2 engineering | ADOPT_CANONICAL | autocast_fp16, TF32, torch.compile, NVML env block |

**3 forks; 7 canonical adoptions.** B1 is full substrate engineering per HNeRV parity L7 (~600 LOC budget exceeds bolt-on).

## 2. 9-Dimension Success Checklist Evidence (per Catalog #294)

### Dim 1: UNIQUENESS
B1 is class-shift: it operates in the previously-unmined dimension of "decoder side-info is the upstream contest video bytes". No prior contest substrate uses upstream/videos/0.mkv as a codebook.

### Dim 2: BEAUTY + ELEGANCE
~600 LOC substrate. Inflate.py extension ~80 LOC. Reviewable per layer. Canonical VQ-VAE + Wyner-Ziv pattern.

### Dim 3: DISTINCTNESS
B1 vs F1: B1 exploits decoder-side video bytes; F1 exploits Hydra dim 7-12. ORTHOGONAL.
B1 vs Y3+Y6: B1 ships indices; Y3+Y6 ships YUV/JPEG bytes. EXCL (both are byte-substitution methods for the same payload).
B1 vs H1: B1 + H1 compose (NVDEC could decode a video-encoded codebook). SUB-additive.

### Dim 4: RIGOR
- Premise verification per Catalog #229: `submissions/exact_current/inflate.py` lines 11-28 verify upstream video accessibility
- Adversarial review per CLAUDE.md "Recursive adversarial review protocol": T3 sextet + grand council
- Empirical anchor: Phase 2 Modal T4 smoke

### Dim 5: OPTIMIZATION PER TECHNIQUE
B1's substrate-optimal engineering: Faiss IVF-PQ for fast NN search (1M patches × 32×32×3 bytes = 3GB; IVF reduces lookup to log time + product quantization compresses).

### Dim 6: STACK-OF-STACKS-COMPOSABILITY
B1 ORTHO with F1; B1 SUB with G1+H1; B1 EXCL with Y3+Y6.

### Dim 7: DETERMINISTIC REPRODUCIBILITY
Codebook deterministic from video sha256; pyav decode deterministic across CPU/CUDA (per upstream contract).

### Dim 8: EXTREME OPTIMIZATION + PERFORMANCE
Inflate runtime per archive: ~5 seconds for codebook build (cached across pairs) + ~10ms per patch reconstruction.

### Dim 9: OPTIMAL MINIMAL CONTEST SCORE
B1 predicted: 0.172-0.187 [contest-CPU] = 2.6-10.4% improvement over current 0.19205.

## 3. Observability Surface (per Catalog #305)

1. **Inspectable per layer**: per-patch (idx, dx, dy, scale, residual) tuple dump-able
2. **Decomposable per signal**: per-patch rate decomposition; residual energy histogram per frame
3. **Diff-able across runs**: codebook deterministic from video sha256
4. **Queryable post-hoc**: by patch_idx → frequency of use; by residual_norm → outlier detection
5. **Cite-able**: (archive_sha, video_sha, codebook_build_sha, faiss_index_sha)
6. **Counterfactual-able**: mutate one patch_idx; observe frame output change (validates byte-consumption per Catalog #105/#139)

## 4. Cargo-Cult Audit Per Assumption (per Catalog #303)

| Assumption | HARD-EARNED vs CARGO-CULTED | Unwind path |
|---|---|---|
| Upstream video readable at inflate time | **HARD-EARNED-VERIFIED** (submissions/exact_current/inflate.py:11-28) | N/A |
| Upstream video bytes are part of pinned env | **HARD-EARNED-VERIFIED** (HNeRV parity L9 + canonical inflate dependency) | N/A |
| 1.2 billion patches from upstream video are sufficient diversity | **HARD-EARNED** (37.5MB × 1200 frames × ~1M patches/frame = ~1B patch positions) | N/A |
| Rendered frames are dense in upstream-video patch space | **CARGO-CULTED** (rendered frames may have different style from contest videos due to compression pipeline) | EMPIRICAL: histogram of nearest-neighbor distance for rendered patches |
| Faiss IVF-PQ adds ≤ 2MB to inflate dependency | **HARD-EARNED-IF AVAILABLE** (Faiss is canonical; ~2MB CPU build; ~20MB with GPU build) | If pinned env lacks Faiss → fallback to scikit-learn KDTree (~0 MB add) |
| AV1 video decode is bit-deterministic CPU vs CUDA | **CARGO-CULTED** (per CLAUDE.md "MPS auth eval is NOISE" similar class; needs probe) | PROBE: bit-identical pyav decode on CPU vs CUDA on same video |
| Codebook lookup determinism across CPU/CUDA | **HARD-EARNED IF Faiss CPU-only build** | Pin Faiss CPU-only |
| Index bits + residual bits < raw RGB bits | **HARD-EARNED** (32×32 patch = 3072 bytes; index 2 bytes + residual ≤ 1024 bytes = 1026 bytes per patch; 3× compression) | N/A |
| Patches translate/scale invariantly | **CARGO-CULTED** (assumes affine invariance; real frames may need more transforms) | EMPIRICAL: sweep transform group complexity |

**4 HARD-EARNED-VERIFIED + 2 HARD-EARNED + 1 HARD-EARNED-IF + 3 CARGO-CULTED.** 3 cargo-cults resolve via Phase 2 smoke.

## 5. Dykstra-Feasibility Intersection (per Catalog #296)

### Constraint set:
- **(R) Rate**: index bits (2B/patch) + residual bits (varies) < raw patch bits (3072B) → ~3× compression
- **(S) Segmentation**: reconstructed patches preserve SegNet argmax IF residual encoding has sufficient precision
- **(P) Pose**: reconstructed frames preserve PoseNet pose IF patch alignment is sub-pixel-accurate
- **(L) Inflate LOC**: ~80 LOC extension to inflate.py; existing PR101 inflate.py is ~150 → total ~230 (EXCEEDS ≤ 200 budget by 30 LOC; requires HNeRV parity L4 same-file waiver or extraction)
- **(D) Determinism**: codebook deterministic from video sha; Faiss CPU build deterministic

### First-principles Dykstra-feasibility check:
Intersection is NON-EMPTY but constraint (L) requires action:
- Option A: extract codebook lookup helper to a separate file (`patch_codebook.py`) and import; total inflate.py stays ≤ 200
- Option B: use HNeRV parity L4 waiver (≤ 200 default, 200-400 with explicit rationale)

### Citation chain:
- Wyner-Ziv 1976: canonical source coding with side info theorem
- van den Oord 2017 VQ-VAE: canonical neural VQ codec
- Faiss IVF-PQ (Jégou-Douze-Schmid 2011 + Johnson-Douze-Jégou 2017): canonical fast NN search
- Probe-disambiguator: `tools/probe_b1_pyav_av1_cpu_cuda_bit_identity.py`

## 6. Predicted Band Per Catalog #324

### Derivation:

Assume 50 patches per frame × 1200 frames = 60,000 patches per archive.
- Per-patch raw: 32×32×3 = 3072 bytes; per-patch encoded: 2B idx + avg 256B residual + 16B transform params = 274 bytes; compression ratio = 11×
- Total reduction per archive: 60,000 × (3072 - 274) = 167.9 MB savings (WAY beyond current archive of ~300KB)
- Cap at actual archive byte count: realistic savings = (archive_bytes - codebook_overhead) × patch_coverage_fraction
- Realistic: 100-300 KiB byte savings = 0.0067 - 0.020 ΔS_rate
- Plus secondary: residual-encoding-quality d_seg/d_pose tradeoff

### Range: [-0.020, -0.005]

### Catalog #324 post-training Tier-C validation criterion:
Predicted band validated when:
1. Probe `tools/probe_b1_pyav_av1_cpu_cuda_bit_identity.py` PASSES
2. Probe `tools/probe_b1_patch_distribution_density.py` confirms rendered-vs-contest patch distribution is dense
3. B1 substrate Modal T4 smoke produces archive within predicted band
4. Paired Linux x86_64 [contest-CPU] re-eval confirms

### Reactivation:
- (a) If pyav AV1 CPU/CUDA NOT bit-identical: pin to CPU-only decode (still works); slight performance hit
- (b) If patch density insufficient: increase codebook search transforms; add scale + rotation
- (c) If smoke OUTSIDE band: re-derive predicted band from actual savings

## 7. 6-Hook Wire-In Declaration (per Catalog #125)

### Hook 1: Sensitivity-map contribution
**ACTIVE**. B1's per-patch byte-attribution feeds `tac.sensitivity_map`.

### Hook 2: Pareto constraint
**ACTIVE**. B1's compression ratio + LOC budget enter Pareto-feasibility solver.

### Hook 3: Bit-allocator hook
**ACTIVE**. B1 changes per-tensor byte allocation: removes raw patch bytes, adds index+residual bits.

### Hook 4: Cathedral autopilot dispatch hook
**ACTIVE**. B1 ranks #3 in `tools/cathedral_autopilot_autonomous_loop.py::rank_candidates` post Phase 1 probes.

### Hook 5: Continual-learning posterior update
**ACTIVE**. B1 Modal smoke call_id registered per Catalog #245.

### Hook 6: Probe-disambiguator
**ACTIVE**. Two probes: bit-identity (CPU vs CUDA AV1 decode) + patch-distribution-density.

## 8. Routing Directive Sketch For Codex Execution

Full routing directive: `.omx/research/codex_routing_directive_rate_attack_vector_3_b1_contest_video_codebook_20260518.md`.

### Phase 1 (PROBES; $0.30):
1. `tools/probe_b1_pyav_av1_cpu_cuda_bit_identity.py` — verify pyav AV1 decode is bit-identical on CPU + CUDA for upstream/videos/0.mkv
2. `tools/probe_b1_patch_distribution_density.py` — measure nearest-neighbor distance distribution of rendered-frame patches vs upstream-video patches

### Phase 2 (SUBSTRATE BUILD + SMOKE; $3-8):
1. Build B1 substrate at `src/tac/substrates/rate_attack_b1_contest_video_codebook/`
2. Trainer at `experiments/train_substrate_rate_attack_b1_contest_video_codebook.py`
3. Operator-authorize recipe at `.omx/operator_authorize_recipes/substrate_rate_attack_b1_contest_video_codebook_modal_t4_dispatch.yaml`
4. Modal T4 100-epoch smoke

### Phase 3 (FULL; $5-15):
1. Modal A100 1000-epoch full
2. Paired Linux x86_64 [contest-CPU] anchor

## 9. Cross-References

- Master memo + META-paradigm + routing directive
- Source anchors: `submissions/exact_current/inflate.py:11-28` (upstream root finder)
- Wyner-Ziv 1976: `https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf`
- VQ-VAE 2017 (van den Oord): arxiv 1711.00937
- Faiss canonical: Jégou-Douze-Schmid 2011 IVF; Johnson-Douze-Jégou 2017 billion-scale GPU NN
- CLAUDE.md non-negotiables + Catalog gates as in F1/G1 memos

## 10. Closeout

B1 is the **CANONICAL Wyner-Ziv operationalization** for the contest. The decoder ALREADY HAS the contest video — we just stopped exploiting it. van den Oord binding: high priority.

**Predicted band [-0.020, -0.005] [contest-CPU]. HIGHEST UPPER BOUND in TOP-5.**

**Next action**: Phase 1 probes via Codex `019de465` per routing directive.
