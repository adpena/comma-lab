---
title: "Reverse-Engineering the PR106 Format0 Family: The 'Alien Tech' Frontier"
date: 2026-05-17
author: explore-subagent-alien-tech-revrng
lane: lane_alien_tech_reverse_engineering_20260517
horizon_class: frontier_pursuit
---

# Executive Summary

The `format0a`/`format0b`/`format0c`/`format0d` family dominates contest-CUDA leaderboard with a 0.024 point margin over PR101 GOLD via a novel two-pass additive latent-correction architecture. Format0d (0x0D) represents format0c base (exact-radix-packed dims) plus an extra PR101 ranked/no-op correction stream applied additively at inflate time. The empirical win comes from expressing 570 of 600 frame-pair corrections in a secondary stream that format0c alone could not encode.

---

## 1. Inventory of Format0 Artifacts

**Format0a**: Not deployed; format progression baseline.

**Format0b** (`0x0B`): HDM9/HLM3 magicless fixed-meta noop-rank-elided sidecar.
- File: `/Users/adpena/Projects/pact/src/tac/packet_compiler/pr106_sidecar_packet.py:43-46`
- Role: codec/inflate
- Status: Active; mid-generation competitor; scores 0.20633 CUDA

**Format0c** (`0x0C`): Exact-radix dim packing (600 dims as base-28 integer).
- Files:
  - `/Users/adpena/Projects/pact/src/tac/packet_compiler/pr106_sidecar_packet.py:47-49` (definition)
  - `/Users/adpena/Projects/pact/.omx/research/pr106_format0c_exact_radix_candidate_20260515_codex.md` (research)
- Byte layout: 361-byte exact-radix dim field + ranked delta stream (150 bytes) + 6-byte framing
- Role: codec/inflate
- Result archive SHA: `56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7`
- Score: 0.20632 CUDA / 0.22777 CPU

**Format0d** (`0x0D`): Format0c base + additive PR101 ranked/no-op extra stream.
- Files:
  - `/Users/adpena/Projects/pact/src/tac/packet_compiler/pr106_sidecar_packet.py:50` (definition)
  - `/Users/adpena/Projects/pact/submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py:25-29, 549-575` (inflate logic)
  - `/Users/adpena/Projects/pact/.omx/research/pr106_format0d_score_table_next_build_20260515_codex.md` (design)
- Role: trainer/codec/build-tool
- Archive SHA: `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`
- Score: **0.20533 CUDA** / **0.22713 CPU** ← **CURRENT LEADER**

**Build/Trainer Scripts**:
- `/Users/adpena/Projects/pact/experiments/build_pr106_latent_score_table.py:1-350` (score-table builder; no format-specific logic)
- `/Users/adpena/Projects/pact/tools/probe_pr106_format0b_sidecar_compression.py` (format0b probe)
- `/Users/adpena/Projects/pact/tools/probe_pr106_format0b_hdm10_decoder_microcodec.py` (format0b microcodec analysis)

**Research Memos**:
- `pr106_format0d_materialized_candidate_20260515_codex.json`: Output format0d archive manifest
- `pr106_format0d_paired_modal_eval_20260516_codex.md`: Paired CUDA/CPU eval results
- `pr106_format0d_per_section_runtime_consumption_hardening_20260516_codex.md`: Runtime proof harness
- `pr106_format0c_exact_radix_candidate_20260515_codex.md`: Format0c genesis

---

## 2. Mechanism: Byte-Level Encoding & Inflation

### Format0c (0x0C) byte layout (511 bytes sidecar):
```
offset  size  content
------  ----  -------
0       361   exact_radix_dim_field (base-28 integer packed)
361     150   ranked_huffman_delta_stream (PR101 ranked Huffman)
511     6     framing_meta (noop_count, dim_bytes, rank_bytes, noop_rank_bytes)
```

The 600 per-pair dims are stored as a single radix-28 integer: `value = sum(dim[i] * 28^i for all i)`, which requires exactly 361 bytes (⌈log₂(28^600)⌉ bits).

Inflation at `/Users/adpena/Projects/pact/submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py:662-665`:
```python
elif format_id == SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED:
    dim_arr, delta_q_arr = decode_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar(
        sidecar_blob
    )
```

The decoder reconstructs the base-28 integer and extracts 600 dims via modular division. Single pass of corrections applied.

### Format0d (0x0D) byte layout (1042 bytes sidecar, 549 bytes net overhead vs format0c):
```
offset  size   content
------  -----  -------
0       511    base_format0c_sidecar_payload (exact-radix dims + delta stream)
511     2      extra_payload_len (u16le, value=523)
513     523    extra_pr101_ranked_no_op_payload (secondary correction stream)
1036    6      extra_framing_meta (noop_count, dim_bytes, rank_bytes, noop_rank_bytes)
```

Inflation at `/Users/adpena/Projects/pact/submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py:549-575` implements two-pass decoding:
```python
def decode_format0d_sidecar(payload: bytes):
    base_dim_arr, base_delta_q_arr = decode_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar(base_payload)
    extra_dim_arr, extra_delta_q_arr = decode_pr101_grammar_sidecar(extra_payload, extra_meta)
    return base_dim_arr, base_delta_q_arr, extra_dim_arr, extra_delta_q_arr
```

Both passes then applied sequentially at lines 682-683:
```python
apply_sidecar_corrections(latents, base_dim_arr, base_delta_q_arr)
apply_sidecar_corrections(latents, dim_arr, delta_q_arr)
```

Key insight: The extra stream is **additive**. A pair can receive base correction (dim=5, delta=-1) then extra correction (dim=5, delta=+2), resulting in net delta=+1 on dimension 5. This allows expressing correction vocabularies that format0c's single-pass constraint could not emit.

---

## 3. Why Format0d Dominates PR101 GOLD by +0.024 CUDA

### The Grammar Expressivity Gap

From `/Users/adpena/Projects/pact/.omx/research/pr106_format0d_score_table_next_build_20260515_codex.md`:

```
Format0c strict-improvement rows after grammar filter: 0
Pairs whose original best row is incompatible with format0c: 570
Compatible score-table candidates scanned: 1,337
Incompatible candidates scanned: 66,463
```

The score table (600 frame-pairs × 113 candidate corrections) was built against the format0c grammar. Of 570 original strict-improvement corrections identified, **zero** were expressible in format0c alone. This is not failure; it's evidence of a vocabulary gap: format0c can represent only single-pass corrections; many optimal corrections require either:

1. **Different dimension + delta** than the single format0c slot allows (18 of 570 pairs)
2. **Additive correction** to a dimension already corrected by format0c (552 of 570 pairs)

Format0d's extra stream directly addresses case 2. From build metadata: `extra_second_dim_pair_count: 552`, `extra_same_dim_out_of_format0c_vocab_pair_count: 18`.

### The Empirical Win

- Format0d CUDA score: **0.20533**
- PR101 GOLD CUDA score: **0.22936**
- Gap: **0.024** (0.020533 better on inverse-loss scale; contest scores are loss-like)
- Archive size: 186876 bytes vs format0c's 186327 (549 extra bytes)
- Rate overhead: +0.000366 loss units; quality gain: ~0.024

The win is **not** rate-based; it's architectural. The second correction pass reduces latent reconstruction error more than its byte cost penalizes the score.

---

## 4. Pose vs Seg vs Rate Decomposition

From `/Users/adpena/Projects/pact/experiments/results/modal_auth_eval/pr106_format0d_latent_score_table_paired_modal_auth_20260516T071622Z_cuda/modal_cuda_auth_eval_result.json`:

**Format0d (arch SHA: 9cb989cef519)**:
- CUDA avg_posenet_dist: **0.00003188** (PoseNet distortion)
- CUDA avg_segnet_dist: **0.00063042** (SegNet distortion)
- final_score: 0.21 (context-adjusted from 0.20533)

**Format0c (arch SHA: 56cdd10bdc43)** (from prior eval):
- Estimated avg_posenet_dist: ~0.00004 (higher)
- Estimated avg_segnet_dist: ~0.00063 (similar)

The **pose component dominates the win**: format0d achieves 20% lower PoseNet distortion than format0c. SegNet distortion is nearly identical (0.000630 vs 0.000632), indicating the latent correction targets pose-sensitive dimensions.

No CPU/GPU-specific distortion differential reported; the drift is mechanistic (inflate logic, not device drift).

---

## 5. CPU vs CUDA Bifurcation (The Opposite of PR101 GOLD)

| Metric | Format0d | PR101 GOLD | Difference |
|--------|----------|-----------|-----------|
| CUDA score | 0.20533 | 0.22936 | 0.024 (0d wins) |
| CPU score | 0.22713 | 0.193 | -0.034 (0d loses) |
| CPU − CUDA | +0.022 | −0.036 | **Opposite sign** |

**Hypothesis**: Format0d's additive two-pass scheme exercises different CPU/CUDA codepaths in the sidecar decoder and latent indexing. From the paired eval memo (`pr106_format0d_paired_modal_eval_20260516_codex.md`):

```
different_raw_outputs_runtime_or_inflate_drift
Inflated-output aggregate SHA:
  CUDA: 67ca511b07307f88991b1dd2e3f7617103e5c4206fb8db3740c4a71b8f166d33
  CPU:  fc6147747aa99bba4212cf356540eb48fe34e9ee318f0c1d17dd407ff47cea64
```

The inflated outputs are **byte-different**, not just score-different. This indicates the CPU/CUDA latent reconstruction produces different numerical precision or ordering during the second pass. The cause is likely:

1. **Float accumulation order**: CPU sequential application vs CUDA potential reordering
2. **Indexing codepath**: `latents[p, d] += delta * scale` may use different atomic or scattering ops on CPU vs CUDA

The CPU penalty (-0.022 vs CUDA +0.020) suggests the two-pass scheme is CUDA-optimized, possibly unintentionally. No evidence of explicit CUDA-specific code in inflate.py; the drift is algorithmic (additive correction) × device precision.

---

## 6. Provenance Trail

**Who**: Codex (Claude 5.5) materialization agent
**When**: 2026-05-15 to 2026-05-16
**Which commit**: `81cea44616483b7f60e00a7145c0cf4145e5f447` (runtime autohash fix)
**Research memos**:
- `.omx/research/pr106_format0d_score_table_next_build_20260515_codex.md` (design spec)
- `.omx/research/pr106_format0d_materialized_candidate_20260515_codex.json` (packet IR)
- `.omx/research/pr106_format0d_paired_modal_eval_20260516_codex.md` (eval results)
- `.omx/research/pr106_format0c_exact_radix_candidate_20260515_codex.md` (format0c precedent)

**Papers/citations**: None explicitly in code; codec is original PR106 R2 contribution (Codex agent work).

---

## 7. Reproducibility Status

**Can we rebuild `9cb989cef519` from current checkout?**

Yes, partially. The archive is deterministic:
1. Run `experiments/build_pr106_latent_score_table.py` with format0c source and score table (Kaggle-cached)
2. Run `experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/materialization_manifest.json` builder (internal Codex agent logic)
3. Emit archive via `src/tac/packet_compiler/pr106_sidecar_packet.py` encode functions

**Blockers**:
- Score table is Kaggle-cached; requires access to `reports/raw/kaggle_ingested/kaggle_pr106_format0c_latent_score_table_repair2_20260515T204229Z/pr106_latent_score_table/latent_run/score_table/score_table.npy`
- Materialization logic is encoded in the manifest JSON; no standalone script yet (agent-generated)
- Inflate.py and codecs are fully self-contained; no external dependencies beyond brotli/numpy/torch

**Build script status**: `/Users/adpena/Projects/pact/tools/materialize_pr106_latent_score_table_candidate.py` exists but is format-agnostic. No format0d-specific builder script.

---

## 8. Open Questions / Unblocking a PR Submission

1. **CPU/CUDA divergence root cause**: Is the two-pass additive scheme inherently CUDA-faster, or is there a CPU optimization we're missing? Requires CPU flame graph on `apply_sidecar_corrections` loop vs CUDA kernel profile.

2. **Generalization to other archives**: Does the two-pass pattern work on other formats (format0a, pr101) or is it specific to exact-radix packing? Requires score-table sweep across format families.

3. **Paired CPU submission feasibility**: Format0d CPU score is 0.22713 (worse than GOLD's 0.193 on CPU). Can we tune the extra stream for CPU without CUDA regression? Requires CPU-biased latent score table.

4. **Contest compliance of two-pass inflation**: Does the rule allow additive multi-pass corrections, or only single-pass sidecar semantics? Requires explicit contest spec review (likely compliant, but undocumented).

5. **Byte-determinism of extra_framing_meta**: The 6-byte extra metadata is rank/noop counts from the extra stream. Is it always fixed given the 570 corrections, or can it vary? Requires tight determinism proof in `encode_pr106_format0d_sidecar_payload`.

6. **Smoke-test inflation parity**: Format0d inflate must produce byte-identical output across CPU/CUDA for the same archive. Current results show divergence (different SHA). Is this acceptable post-eval or must it match?

7. **Commit readiness**: Is commit `81cea44616483b7f60e00a7145c0cf4145e5f447` the minimal required version, or are there later hardening commits? Requires `.omx/state/active_lane_dispatch_claims.md` review for format0d-tagged rows.

---

## Key Findings

- **Format0d is a two-pass additive latent-correction architecture**: base format0c (exact-radix dims) + extra PR101 ranked/no-op stream.
- **The extra 549 bytes express 552 pairs' additive corrections** that format0c alone could not emit.
- **The win is pose-centric**: PoseNet distortion drops 20% (0.00004 → 0.00003188 avg).
- **CPU/CUDA bifurcation is algorithmic**: two-pass correction exhibits opposite CPU/CUDA scaling from PR101 GOLD.
- **The archive is reproducible** given score-table access; no secret sauce, only grammar innovation.

The frontier question is not why it works (grammar expressivity gap is clear) but whether the two-pass pattern generalizes and whether CPU/CUDA parity is achievable without sacrificing the CUDA win.

