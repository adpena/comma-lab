---
schema: codex_routing_directive_v1
directive_id: codex_routing_directive_rate_attack_vector_3_b1_contest_video_codebook_20260518
target_subagent: codex_019de465
routing_date: "2026-05-18"
parent_design_memo: rate_attack_vector_3_b1_contest_video_codebook_design_memo_20260518
parent_master_memo: rate_attack_43_vectors_meta_paradigm_deep_research_20260518
meta_paradigm_anchor: structural_information_not_shipped_meta_paradigm_unification_20260518
vector_id: B1
priority: TOP-3
council_verdict: PROCEED_WITH_REVISIONS_PROBES_REQUIRED
binding_revisions:
  - "van den Oord: 32x32 patches + 2048-entry codebook initial sweep"
  - "Fridrich: AV1 decode CPU/CUDA bit-identity probe before substrate dispatch"
operator_approved_gpu_budget_usd: 12.00
operator_approved_gpu_budget_phase_breakdown:
  phase_1_probes: 0.30
  phase_2_substrate_smoke: 4.00
  phase_3_full_run: 8.00
write_scope_for_codex: |
  src/tac/substrates/rate_attack_b1_contest_video_codebook/  (NEW package)
  experiments/train_substrate_rate_attack_b1_contest_video_codebook.py  (NEW trainer)
  submissions/rate_attack_b1/  (NEW submission)
  tools/probe_b1_pyav_av1_cpu_cuda_bit_identity.py  (NEW probe)
  tools/probe_b1_patch_distribution_density.py  (NEW probe)
  scripts/remote_lane_substrate_rate_attack_b1_contest_video_codebook.sh  (NEW driver)
  .omx/operator_authorize_recipes/substrate_rate_attack_b1_contest_video_codebook_modal_t4_dispatch.yaml  (NEW recipe)
  src/tac/tests/test_rate_attack_b1_contest_video_codebook.py  (NEW tests)
write_scope_excludes:
  - "Anything in PRIMARY research subagent scope"
  - "Anything in ADVERSARIAL sister subagent scope"
---

# Codex Routing Directive — Rate-Attack Vector 3: B1 Contest-Video-As-Codebook

**Target Codex subagent**: `019de465`
**Priority**: TOP-3 (HIGHEST UPPER BOUND in TOP-5; canonical Wyner-Ziv operationalization)
**META-paradigm**: SINS — decoder ALREADY HAS upstream contest video bytes (37.5 MB readable at inflate per `submissions/exact_current/inflate.py:11-28`). Use them as VQ codebook.

## 0. PRE-FLIGHT (per Phase 1 F1/G1 routing directives + Catalog #206/229)

1. Read CLAUDE.md + AGENTS.md + MEMORY.md top-50
2. Read parent design memo: `.omx/research/rate_attack_vector_3_b1_contest_video_codebook_design_memo_20260518.md`
3. Read parent master memo + META-paradigm
4. Read source anchor: `submissions/exact_current/inflate.py:11-28` (find_upstream_root)
5. Read Wyner-Ziv 1976 PDF reference: `https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf`
6. Read van den Oord VQ-VAE 2017: arxiv 1711.00937
7. Read Faiss IVF-PQ canonical: Jégou-Douze-Schmid 2011 + Johnson-Douze-Jégou 2017
8. Lane registry check: `lane_rate_attack_b1_contest_video_codebook_substrate_20260518` pre-registered at L0

## 1. Phase 1 — PROBES ($0.30; MODAL T4)

### 1.1 `tools/probe_b1_pyav_av1_cpu_cuda_bit_identity.py` (~120 LOC)

**Goal**: Verify pyav AV1 decode is bit-identical on CPU + CUDA for upstream/videos/0.mkv.

**Pseudocode**:
```python
def probe_av1_bit_identity():
    import av
    container_cpu = av.open("upstream/videos/0.mkv")
    # Decode 100 frames on CPU
    cpu_frames = [frame.to_ndarray(format='rgb24') for frame in container_cpu.decode(video=0)[:100]]

    container_cuda = av.open("upstream/videos/0.mkv", options={'hwaccel': 'cuda'})
    # Decode 100 frames on CUDA
    cuda_frames = [frame.to_ndarray(format='rgb24') for frame in container_cuda.decode(video=0)[:100]]

    bit_identical = all(np.array_equal(cpu_frames[i], cuda_frames[i]) for i in range(100))

    # If NOT bit-identical: measure max ULP difference
    if not bit_identical:
        max_diff = max(np.abs(cpu_frames[i].astype(int) - cuda_frames[i].astype(int)).max() for i in range(100))
        verdict = f"NOT_BIT_IDENTICAL_max_diff_{max_diff}"
    else:
        verdict = "BIT_IDENTICAL"

    # Register probe outcome per Catalog #313
    from tac.probe_outcomes_ledger import register_probe_outcome
    register_probe_outcome(
        probe_id=f"b1_pyav_av1_cpu_cuda_bit_identity_{int(time.time())}",
        substrate_id="rate_attack_b1_contest_video_codebook",
        recipe_path="substrate_rate_attack_b1_contest_video_codebook_modal_t4_dispatch.yaml",
        verdict="PROCEED" if bit_identical else "PROCEED_WITH_REVISIONS_PIN_DECODER",
        rationale=f"AV1 decode bit-identity probe: {verdict}",
        status="advisory" if bit_identical else "blocking",
        event_type="adjudicated",
    )
```

**Gates**:
- If BIT_IDENTICAL → Phase 1.2
- If NOT BIT_IDENTICAL → recipe requires pinning to ONE decoder (CPU libaom OR NVDEC) via Catalog #205 `PACT_INFLATE_DEVICE` env var

### 1.2 `tools/probe_b1_patch_distribution_density.py` (~150 LOC)

**Goal**: Measure nearest-neighbor distance distribution of rendered-frame patches vs upstream-video patches.

**Pseudocode**:
```python
def probe_patch_density():
    # Extract 32x32 patches from upstream/videos/0.mkv
    upstream_patches = extract_patches_from_video("upstream/videos/0.mkv", patch_size=32)
    # ~ 1B patches; subsample to 2M for codebook construction
    upstream_codebook = build_faiss_ivf_pq_index(upstream_patches, nlist=2048, m=8)

    # Extract 32x32 patches from a RENDERED frontier archive (PR101)
    rendered_frames = inflate_and_decode_to_rgb("submissions/pr101_*/archive.zip")
    rendered_patches = extract_patches(rendered_frames, patch_size=32)

    # For each rendered patch, find nearest upstream patch + distance
    distances = upstream_codebook.search(rendered_patches, k=1)[0]

    # Histogram
    p50 = np.percentile(distances, 50)
    p95 = np.percentile(distances, 95)
    p99 = np.percentile(distances, 99)

    # Verdict: if p50 < threshold → upstream codebook is DENSE for rendered patches
    threshold = 100  # L2 distance threshold (TUNE empirically)
    verdict = "DENSE" if p50 < threshold else "SPARSE"

    register_probe_outcome(
        probe_id=f"b1_patch_distribution_density_{int(time.time())}",
        substrate_id="rate_attack_b1_contest_video_codebook",
        recipe_path="substrate_rate_attack_b1_contest_video_codebook_modal_t4_dispatch.yaml",
        verdict="PROCEED" if verdict == "DENSE" else "PROCEED_WITH_REVISIONS_INCREASE_CODEBOOK_TRANSFORMS",
        rationale=f"Patch distribution p50={p50}, p95={p95}, p99={p99}, threshold={threshold} → {verdict}",
        status="advisory",
        event_type="adjudicated",
    )
```

**Gates**:
- If DENSE → Phase 2
- If SPARSE → recipe needs MORE TRANSFORMS (scale + rotation + multi-scale patches) OR alternative side info

## 2. Phase 2 — SUBSTRATE BUILD + SMOKE ($3-8; MODAL T4)

### 2.1 Substrate package `src/tac/substrates/rate_attack_b1_contest_video_codebook/`

Files:
- `__init__.py` + register_substrate(SubstrateContract(...)) per Catalog #241/#242
- `architecture.py` — VQ encoder + codebook builder
- `codebook.py` — Faiss IVF-PQ wrapper
- `archive.py` — B1 grammar (codebook params + per-patch (idx, dx, dy, scale, residual) tuples)
- `score_aware_loss.py` — ADOPT canonical `score_pair_components`
- `inflate_runtime.py` — codebook reconstruction at inflate time
- `tests/`

### 2.2 Trainer `experiments/train_substrate_rate_attack_b1_contest_video_codebook.py`

ADOPT all canonical primitives per F1 routing directive §2.2. Additionally:
- `--codebook-size` (default 2048 per van den Oord binding)
- `--patch-size` (default 32 per van den Oord binding)
- `--num-transforms` (default 4: identity + 3 scales {0.5, 1.0, 2.0})

### 2.3 Submission `submissions/rate_attack_b1/`

- `inflate.py` (~180 LOC; HNeRV parity L4 waiver may be needed if ≥ 200): ADOPT canonical inflate device + reads upstream/videos/0.mkv (already pinned upstream) + Faiss IVF-PQ codebook lookup + reconstruction loop
- `inflate.sh` per Catalog #146

### 2.4 Recipe per F1 routing directive §2.4 template

Customizations:
- `substrate_id: rate_attack_b1_contest_video_codebook`
- `min_vram_gb: 14`
- `predicted_band: "[-0.020, -0.005]"`
- `required_input_files: video_upstream_main: {flag: "--upstream-video", default_path: "upstream/videos/0.mkv", required_input_file: true}`
- `TIER_1_EXTRA_MOUNT_PATHS: ["upstream/videos/0.mkv"]` per Catalog #152

### 2.5 Driver per F1 routing directive §2.5 template

Customizations:
- Canonical NVML env block per Catalog #244
- Multi-candidate path resolution for upstream video per Catalog #152 Wave 2 (already in submissions/exact_current/inflate.py pattern)

### 2.6 Dispatch via canonical `tools/operator_authorize.py`

Per F1 routing directive §2.6.

### 2.7 Paired Linux x86_64 [contest-CPU] anchor per CLAUDE.md

## 3. Phase 3 — FULL ($5-8; CONDITIONAL on Phase 2 smoke validating)

Per F1 routing directive §3.

## 4. Discipline (NON-NEGOTIABLE)

Per F1 routing directive §4. Additional:
- pyav AV1 decode determinism: pin to ONE decoder per Phase 1.1 probe verdict
- Faiss IVF-PQ: prefer CPU-only build for determinism (~2 MB add); GPU build optional
- HNeRV parity L9 dependency closure: ALL deps (pyav, faiss, libaom) MUST be in pinned upstream

## 5. Cross-References

- Parent design memo + parent master memo + META-paradigm
- Sister B1 source anchor: `submissions/exact_current/inflate.py:11-28`
- Wyner-Ziv 1976 + van den Oord VQ-VAE 2017 + Faiss IVF-PQ canonical references
- CLAUDE.md non-negotiables: all from F1 routing directive
- Catalog gates: all from F1 routing directive

## 6. Closeout

B1 has the HIGHEST UPPER BOUND in TOP-5 (-0.020). Canonical Wyner-Ziv. van den Oord binding: high priority.

**Predicted band [-0.020, -0.005] [contest-CPU]. Decoder has upstream video for FREE.**

**Next action**: Phase 1 probes via Codex `019de465`. Probe 1 ($0.30 Modal T4) + Probe 2 ($0 local).
