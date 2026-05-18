---
schema: pact_design_memo_v1
memo_id: rate_attack_vector_5_h1_nvdec_hardware_decode_design_memo_20260518
review_date: "2026-05-18"
lane_id: lane_rate_attack_h1_nvdec_hardware_decode_substrate_20260518
parent_master_memo: rate_attack_43_vectors_meta_paradigm_deep_research_20260518
meta_paradigm_anchor: structural_information_not_shipped_meta_paradigm_unification_20260518
vector_id: H1
vector_name: "NVDEC hardware video decode (T4 has 1 NVDEC engine; ship AV1/HEVC bytes; decoder pays 0 compute)"
horizon_class: frontier_breaking
predicted_band_validation_status: pending_post_training
score_claim: false
promotion_eligible: false
research_only: true
write_scope: ".omx/research only"
canonical_frontier_anchor:
  contest_cpu: "0.19205 [contest-CPU]"
predicted_delta_band_contest_cpu: "[-0.025, -0.008]"
council_tier_assignment: T3_full_grand_council
target_modes:
  - contest_exact_eval
  - contest_generalized
deployment_target: t4_contest_runtime
hardware_substrate: linux_x86_64_t4_with_nvdec
---

# TOP-5 Design Memo — Vector H1: NVDEC Hardware Video Decode

**Master memo**: `.omx/research/rate_attack_43_vectors_meta_paradigm_deep_research_20260518.md`
**META-paradigm**: SINS
**Lane**: `lane_rate_attack_h1_nvdec_hardware_decode_substrate_20260518` L0

## 0. Executive Summary

**HARD-EARNED hardware anchor**: T4 has 1× NVDEC engine supporting H.264/H.265/AV1/VP8/VP9/MPEG-4 + 1× NVENC encoding engine (per NVIDIA NVENC/NVDEC SDK + T4 spec). Pinned upstream includes `pyav` which supports NVDEC backend via FFmpeg's `-hwaccel cuda`.

**CARGO-CULTED until verified**: NVDEC accessibility at INFLATE time on the contest worker. Requires probe.

**H1 exploit**: ship a tiny AV1/HEVC-encoded "residual video" (e.g., 30 KiB for 1200 frames) + use NVDEC to decode at inflate time. Decoder pays ZERO compute cost (hardware-accelerated). Inflate.py LOC: ~30 LOC for pyav + NVDEC + CPU fallback.

**Predicted ΔS**: [-0.025, -0.008] — **HIGHEST UPPER BOUND** in TOP-5 (modern AV1 achieves 50% smaller than HEVC; HEVC 50% smaller than H.264; cumulative compression 75-95% for typical video content).

**Council verdict**: T3 PROCEED_WITH_REVISIONS (Carmack binding: probe NVDEC availability before substrate dispatch).

## 1. Canonical-vs-unique Decision Per Layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| pyav decode | ADOPT_CANONICAL | Pinned upstream dependency per HNeRV parity L9 |
| NVDEC backend (FFmpeg hwaccel cuda) | ADOPT_CANONICAL_IF_AVAILABLE | If pinned env has FFmpeg with hwaccel cuda built; needs probe |
| CPU AV1 fallback (libaom) | ADOPT_CANONICAL | Pinned upstream libaom |
| AV1 encoder (compress-time) | ADOPT_CANONICAL (NVENC if available; CPU libaom otherwise) | Compress-time not scored |
| Catalog #205 inflate device selector | ADOPT_CANONICAL | Operator-pinnable via PACT_INFLATE_DEVICE env var |
| Archive grammar | **FORK_BECAUSE_SUBSTRATE_OPTIMAL** | NEW: AV1 video bytes as primary payload section + traditional bytes section for non-video-encodable content |
| Inflate runtime | ADOPT_CANONICAL + extension | ~30 LOC for pyav.open + NVDEC backend selection + CPU fallback |
| Residual encoding (for video-incompatible bytes) | ADOPT_CANONICAL (Brotli) | Existing canonical |
| Score-aware loss | ADOPT_CANONICAL | Existing canonical |
| EMA + eval_roundtrip + Tier-1/2 engineering | ADOPT_CANONICAL | All existing canonical |

## 2. 9-Dim Checklist (per Catalog #294)

- Dim 1 UNIQUENESS: H1 is class-shift (hardware-codec category; first contest substrate to exploit NVDEC for archive bytes)
- Dim 2 BEAUTY: ~200 LOC substrate; ~30 LOC inflate extension; reviewable
- Dim 3 DISTINCTNESS: H1 vs F1: orthogonal; H1 vs B1: SUB (could combine via NVDEC-decoded codebook video); H1 vs Y3+Y6: SUB (NVDEC could decode JPEG)
- Dim 4 RIGOR: Premise verification + NVDEC SDK citation + Carmack binding probe
- Dim 5 OPTIMIZATION: NVDEC hardware decode at ~1 GB/s throughput; AV1 30-50% smaller than HEVC
- Dim 6 COMPOSABILITY: H1 SUB with B1+Y3+Y6+G1; ORTHO with F1
- Dim 7 REPRODUCIBILITY: AV1 decode is bit-deterministic per AV1 spec (NVDEC vs libaom must produce identical output OR pin to one decoder)
- Dim 8 OPTIMIZATION: T4 NVDEC throughput ≥ 250 fps at 4K; sub-second decode for entire archive
- Dim 9 OPTIMAL SCORE: predicted [0.167, 0.184] [contest-CPU] = 4.2-13.0% improvement

## 3. Observability Surface (per Catalog #305)

1. Inspectable: per-frame AV1 bitstream dump-able
2. Decomposable: per-frame entropy; per-frame bitrate
3. Diff-able: NVDEC decode output vs CPU libaom decode output (bit-identity check)
4. Queryable: per-frame quality (PSNR, SSIM); per-archive total bitrate
5. Cite-able: (archive_sha, av1_codec_version, nvdec_driver_version, libaom_version)
6. Counterfactual-able: re-encode with different bitrate; observe score curve

## 4. Cargo-Cult Audit (per Catalog #303)

| Assumption | Verdict |
|---|---|
| T4 has NVDEC engine | **HARD-EARNED** (NVIDIA T4 spec: 1× NVDEC) |
| pyav is pinned in upstream | **HARD-EARNED-VERIFIED** (submissions/exact_current/inflate.py uses `import av`) |
| pyav supports NVDEC backend at inflate time | **CARGO-CULTED-PENDING-VERIFICATION** (needs `pip list` + `pyav.codec.Codec('av1', 'r').is_hardware`) |
| NVDEC AV1 decode is bit-identical to CPU libaom decode | **CARGO-CULTED** (HARDWARE codec quirks per CLAUDE.md MPS-noise lineage; needs probe) |
| AV1 ships 30-50% smaller than HEVC | **HARD-EARNED** (AOMedia AV1 spec + DCVC-FM 2024 benchmark) |
| HEVC ships 50% smaller than H.264 | **HARD-EARNED** (ITU-T H.265 spec) |
| Rendered frames are AV1-compressible (low entropy) | **HARD-EARNED** (rendered frames have spatial+temporal redundancy; classic video codec sweet spot) |
| 30 KiB AV1 bytes can carry 1200 frames at acceptable quality | **CARGO-CULTED-PENDING-EMPIRICAL** (depends on per-frame entropy; needs measurement) |

## 5. Dykstra-Feasibility (per Catalog #296)

- (R) AV1 bytes are 1-5% of equivalent uncompressed; massive rate-term savings IF the entire payload can be video-encoded
- (S) AV1 lossy compression preserves SegNet argmax at sufficient bitrate (CRF 30-45 range typical for low-bitrate-acceptable-quality)
- (P) AV1 preserves PoseNet pose at sufficient bitrate
- (L) ~30 LOC inflate extension; total ~180 LOC ≤ 200
- (D) AV1 decode determinism CPU vs CUDA REQUIRES verification

## 6. Predicted Band (per Catalog #324)

Derivation:
- Worst-case: rendered frames are AV1-compressed at ~50KiB total (vs ~300KiB raw archive); savings ~250KiB × 6.657e-7 = 0.167 ΔS — TOO LARGE; can't claim
- Realistic: rendered residuals are AV1-compressible at ~30-50% of current encoding; savings ~120KiB - 150KiB × 6.657e-7 = 0.080-0.100 ΔS — STILL TOO LARGE; can't claim
- Conservative: integration friction + AV1 inflate parsing overhead + scorer-quality preservation requires bitrate above AV1 minimum; achievable savings ~12-37 KiB = 0.008-0.025 ΔS

Range: [-0.025, -0.008] (conservative because integration risk is real)

## 7. 6-Hook Wire-In (per Catalog #125)

All 6 ACTIVE.

## 8. Routing Directive Sketch

Full directive: `.omx/research/codex_routing_directive_rate_attack_vector_5_h1_nvdec_hardware_decode_20260518.md` (DEFERRED in this wave; per master memo §6 TOP-3 = F1+G1+B1).

### Phase 1 (PROBES; $0.30):
1. `tools/probe_nvdec_availability_on_contest_t4.py` — Modal T4 smoke; verify pyav AV1 NVDEC backend accessible
2. `tools/probe_av1_decode_cpu_cuda_bit_identity.py` — verify NVDEC vs libaom bit-identity (likely will FAIL; will require pinning to one decoder)

### Phase 2 (SUBSTRATE BUILD; $1-2):
1. Build H1 substrate; Modal T4 smoke with AV1-encoded residual video

## 9. Cross-References

- Master memo + META-paradigm
- NVIDIA NVDEC SDK: `https://developer.nvidia.com/nvidia-video-codec-sdk`
- AV1 spec: AOMedia
- DCVC-FM 2024 (Lu-Ouyang-Xu-Zhang): canonical neural compression matching VVC
- VVC/H.266 spec: ITU-T H.266 / ISO/IEC 23090-3 (2020)
- Catalog #205 inflate device selector

## 10. Closeout

H1 has the HIGHEST UPPER BOUND of TOP-5 (-0.025) but HIGHEST UNCERTAINTY (NVDEC availability CARGO-CULTED). Carmack binding: probe first.

**Predicted band [-0.025, -0.008] [contest-CPU]. CONDITIONAL on NVDEC probe PASS.**

**Next action**: Phase 1 NVDEC availability probe per Codex `019de465` (DEFERRED in this wave; routing directive TOP-3 = F1+G1+B1).
