# Posterior recovery landscape — 2026-05-13

## TL;DR

`tools/bulk_backfill_anchors_into_posterior.py` ingested **17 new anchors** into
`.omx/state/continual_learning_posterior.json` (25 → **42 accepted**). All
writes flowed through `tac.continual_learning.posterior_update_locked()` per
Catalog #128 (fcntl `LOCK_EX` on `.omx/state/.continual_learning.lock`); custody
verdicts respected per Catalog #127.

- Total artifacts discovered: **279** `contest_auth_eval.json` (across 58
  result-dir families).
- Already in posterior: 108 (duplicates of the 25 historical anchors — same
  sha+axis re-evaluated).
- **Promotable orphans found: 21** (13 CPU + 8 CUDA).
- Custody-refused: 225 (156 `macos_substrate` + 69 `advisory_grade` — correct
  per CLAUDE.md "macOS-CPU is NEVER 1:1 contest-compliant").
- Backfill commit: 17 accepted, 4 idempotent-duplicate refusals.
- Audit log: `experiments/results/bulk_backfill_20260513T230000Z/audit.jsonl`
- Summary JSON: `experiments/results/bulk_backfill_20260513T230000Z/summary.json`

## Per-axis Pareto frontier (post-backfill)

### CUDA axis — top 10 (lowest score)

| Score | Archive bytes | Architecture | sha (12) |
|---:|---:|---|---|
| **0.206380** | 186,423 | hnerv_hlm1_fixed_latent_recode | 8801845d5099 |
| 0.206426 | 186,492 | hnerv_hdm4_q_brotli_split | 218ae16f3f13 |
| 0.206508 | 186,615 | pr106_r2_lowlevel_hdm3_sidecar_pr101 | 8cc7e3b21a5f |
| 0.206517 | 186,629 | pr106_r2_pr101_grammar_lowlevel_repack_151b | 287e6edc6128 |
| 0.206618 | 186,780 | lane_pr106_latent_sidecar_r2_pr101_grammar | c48631e11a9b |
| 0.206634 | 186,832 | lane_c3_residual_pr106_sidecar | eafb1a027f70 |
| 0.206634 | 186,832 | lane_wavelet_residual_pr106_sidecar | ed90a2250e94 |
| 0.206634 | 186,832 | lane_c3_residual_pr106_sidecar_l2_sparse_aware | 8e61ff2d5a42 |
| 0.206634 | 186,832 | lane_cool_chic_residual_pr106_sidecar | d48600da99ba |
| 0.206634 | 186,832 | lane_coord_mlp_residual_pr106_sidecar | 01df6f12e562 |

**CUDA frontier saturates at ~0.20663** for the PR106-sidecar family — six
distinct sidecar variants (c3, wavelet, cool_chic, coord_mlp, siren, sparse_aware)
all land at identical score. Suggests the sidecar payload bytes don't move
score-affecting kernels; either `no_op_detector` would refuse, or the variants
share an identity bytestream that just got re-classified.

### CPU axis — top 11 (lowest score)

| Score | Archive bytes | Architecture | sha (12) |
|---:|---:|---|---|
| **0.192848** | **178,262** | **hnerv_ft_microcodec (A1)** | 87ec7ca5f2f3 |
| 0.227827 | 186,423 | hnerv_hlm1_fixed_latent_recode | 8801845d5099 |
| 0.227875 | 186,492 | pr106_r2_lowlevel_hdm4 | 218ae16f3f13 |
| 0.227964 | 186,629 | pr101_lossy_coarsening | 287e6edc6128 |
| 0.228066 | 186,780 | lane_pr106_latent_sidecar_r2_pr101_grammar | c48631e11a9b |
| 0.228092 | 186,822 | lane_pr106_latent_sidecar_r2 | 7f926bc3e213 |
| 0.228102 | 186,832 | lane_c3_residual_pr106_sidecar | eafb1a027f70 |
| 0.228102 | 186,832 | lane_wavelet_residual_pr106_sidecar | ed90a2250e94 |
| 0.228680 | 186,808 | lane_pr106_latent_sidecar_r1 | 947b85e8a69d |
| 0.229666 | 185,578 | pr103_on_pr106 | ec0890c2d231 |
| 1.037583 | 694,074 | unknown__contest_cpu_eval-lane_g_v3 | 9b20bdfca246 |

**The A1 hnerv_ft_microcodec at 0.192848 [contest-CPU-1to1] is the lone
sub-0.20 anchor on the CPU axis** — and the only architecture in the entire
posterior that beats the public PR102 medal-band score on CPU.

## Ship-readiness classification (per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable)

### Dual-axis (ship-ready) — 10 archives

Same `archive_sha256` measured on BOTH `[contest-CPU-1to1]` (Linux x86_64 GHA)
AND `[contest-CUDA]` (NVIDIA T4/A100/4090 Modal/Vast):

| Architecture | Bytes | CPU | CUDA | gap (CUDA−CPU) |
|---|---:|---:|---:|---:|
| **hnerv_ft_microcodec (A1)** | **178,262** | **0.192848** | 0.226352 | **+0.0335** |
| pr106_r2 PR101 grammar lowlevel repack 151b | 186,629 | 0.227964 | 0.206517 | −0.0214 |
| hnerv_hdm4_q_brotli_split | 186,492 | 0.227875 | 0.206426 | −0.0214 |
| hnerv_hlm1_fixed_latent_recode | 186,423 | 0.227827 | 0.206380 | −0.0214 |
| lane_pr106_latent_sidecar_r2_pr101_grammar | 186,780 | 0.228066 | 0.206618 | −0.0214 |
| lane_pr106_latent_sidecar_r2 | 186,822 | 0.228092 | 0.206646 | −0.0214 |
| lane_pr106_latent_sidecar_r1 | 186,808 | 0.228680 | 0.207394 | −0.0213 |
| lane_c3_residual_pr106_sidecar | 186,832 | 0.228102 | 0.206634 | −0.0215 |
| lane_wavelet_residual_pr106_sidecar | 186,832 | 0.228102 | 0.206634 | −0.0215 |
| pr103_on_pr106 | 185,578 | 0.229666 | 0.208983 | −0.0207 |

**Two regimes are visible**:

1. **A1 hnerv_ft_microcodec**: CPU < CUDA by **+0.034** (CPU is BETTER than CUDA).
   This matches the PR102/103 leaderboard pattern (CPU < CUDA).
2. **PR106-family + HNeRV recode + sidecars**: CPU > CUDA by a uniform
   **−0.021** (CUDA is BETTER than CPU). This is the OPPOSITE of the public
   leaderboard pattern. The CUDA-CPU gap is essentially constant across 9
   different PR106-family variants (range −0.0207 to −0.0215, σ ≈ 0.0003).

The −0.021 gap is roughly 5× smaller in magnitude than PR102's −0.033 leaderboard
gap, but in the wrong direction for medal-band ranking (the contest ranks by
CPU, so PR106-family scores ~0.228 [contest-CPU-1to1] is **NOT competitive**
with PR102's 0.195).

### Single-axis CUDA-only (NOT ship-ready) — 21 archives

Need CPU GHA Linux x86_64 pairing before any submission claim. Top 5 by score:

| Score (CUDA) | Bytes | Architecture | sha (12) |
|---:|---:|---|---|
| 0.206508 | 186,615 | pr106_r2_lowlevel_hdm3_sidecar_pr101_runtime | 8cc7e3b21a5f |
| 0.206634 | 186,832 | lane_c3_residual_pr106_sidecar_l2_sparse_aware | 8e61ff2d5a42 |
| 0.206634 | 186,832 | lane_cool_chic_residual_pr106_sidecar | d48600da99ba |
| 0.206634 | 186,832 | lane_coord_mlp_residual_pr106_sidecar | 01df6f12e562 |
| 0.206634 | 186,832 | lane_siren_residual_pr106_sidecar | f373b308b080 |

Plus 16 more (pr101_lossy_coarsening cluster, pr103_arithmetic_coding cluster,
unknown_eval_work, etc.).

### Single-axis CPU-only (NOT ship-ready) — 1 archive

| Score (CPU) | Bytes | Architecture | sha (12) |
|---:|---:|---|---|
| 1.037583 | 694,074 | unknown__contest_cpu_eval-lane_g_v3 | 9b20bdfca246 |

The legacy lane_g_v3 1.05 anchor on CPU axis without a CUDA twin in the
posterior.

## Under-tested substrate bands

By byte-target band:

- **186,500-186,900 (PR106-family band)**: 10 archives, dual-axis → well covered.
- **186,000-186,500 (HNeRV recode + pr106_q sweep)**: 4 archives, mixed.
- **178,000-178,500 (A1 hnerv_ft_microcodec band + pr101_lossy)**: only 1 archive
  (A1) has dual-axis; 3 pr101_lossy variants are CUDA-only. **CRITICAL GAP**.
- **185,500-186,000 (pr103-on-pr106)**: 1 archive, dual-axis.
- **694,074 bytes**: legacy lane_g_v3 only; not a frontier band.

## Top 5 highest-leverage cheap-dispatch suggestions

Per CLAUDE.md "Submission auth eval BOTH CPU AND CUDA on 1:1 contest-compliant
hardware" + the 178,000-byte band gap:

1. **GHA CPU eval the 3 pr101_lossy_coarsening CUDA-only archives** (~178K bytes,
   CUDA scores 0.226-0.227). If their CPU axis lands sub-0.20 (matching A1's
   pattern), they immediately become ship-ready dual-axis candidates. Cost:
   3× ~$0 GHA Actions runs.

2. **GHA CPU eval the 5 PR106-sidecar L2-sparse-aware/cool_chic/coord_mlp/siren
   CUDA-only archives** (186,832 bytes each, identical CUDA score 0.206634).
   Confirms whether the sidecar variants drift on CPU axis or all collapse to
   the same ~0.228 plateau. Cost: 5× ~$0 GHA runs.

3. **GHA CPU eval pr103_arithmetic_coding cluster** (178,200 bytes, 0.227-0.230
   CUDA). The 178K-byte band is where the medal-band lives; need CPU paired
   anchors.

4. **CUDA T4 eval the lane_g_v3 1.05 CPU-only archive** to close the legacy
   single-axis row. Cost: ~$0.10 Modal T4.

5. **Probe the +0.022 vs −0.034 CPU/CUDA-gap inversion**: the PR106-family is
   uniformly CPU-WORSE-than-CUDA, while A1 hnerv_ft_microcodec and PR102 are
   CPU-BETTER-than-CUDA. This suggests the PR106-family decoder has CPU-vs-CUDA
   numerics divergence in either renderer or scorer-derived metrics that the
   A1 microcodec does not. A 2x2 decoder/network diagnostic on PR106 vs A1
   on the same hardware could explain the mechanism. Cost: ~$0.50 Modal A100.

## Custody-refusal taxonomy (per Catalog #127)

| Refused class | Count | Meaning |
|---|---:|---|
| `macos_substrate` | 156 | macOS Apple Silicon CPU eval — NEVER 1:1 contest-compliant. Per CLAUDE.md "Submission auth eval" non-neg, must be tagged `[macOS-CPU advisory]`, NOT promotable. |
| `advisory_grade` | 69 | Tagged `evidence_grade="MPS-research-signal"` or `[macOS-CPU advisory]` — explicit non-promotable signal. |

These 225 refusals are **correct fail-closed behavior** — the validator did its
job. Memory of these refusals is preserved in the audit JSONL so future ranking
loops can use them as advisory priors without confusing them with authoritative
anchors.

## Wire-in declarations (CLAUDE.md "Subagent coherence-by-default" 6 hooks)

1. **Sensitivity-map contribution**: 17 new anchors expand the per-architecture
   drift posterior estimation surface that
   `tac.sensitivity_map.*` consumes; in particular the +0.022 vs −0.034
   CPU/CUDA-gap split (PR106-family vs A1) is now empirically anchored as
   per-class drift signal.
2. **Pareto constraint**: 21 single-axis CUDA-only archives at the 178K-byte
   band and the 186K-byte band tighten the Pareto frontier intersection
   `F_seg ∩ F_pose ∩ F_rate` for the dual-eval-required submission feasibility
   region; explicit single-axis blockers documented above.
3. **Bit-allocator hook**: N/A — this is a posterior-state recovery, no per-tensor
   importance signal changes.
4. **Cathedral autopilot dispatch hook**: 5 cheap-dispatch suggestions above
   feed `tools/cathedral_autopilot_autonomous_loop.py` candidate ranking with
   the 178K-byte band as priority ROI.
5. **Continual-learning posterior update**: this IS the posterior update — 17
   anchors added via `posterior_update_locked` (Catalog #128).
6. **Probe-disambiguator**: the +0.022 vs −0.034 CPU/CUDA-gap inversion (PR106-
   family vs A1) is exactly the "2+ defensible interpretations" pattern that
   warrants a `tools/probe_pr106_vs_a1_cpu_cuda_gap_disambiguator.py` to
   isolate decoder-numerics drift vs scorer-numerics drift vs YUV6-preprocess
   drift.

## Cross-references

- `tools/bulk_backfill_anchors_into_posterior.py` — canonical helper used.
- `tac.continual_learning.posterior_update_locked` — Catalog #128 lock.
- `tac.continual_learning.validate_custody_verdict` — Catalog #127 typed validator.
- `experiments/results/bulk_backfill_20260513T230000Z/audit.jsonl` — per-anchor audit log.
- `experiments/results/bulk_backfill_20260513T230000Z/summary.json` — structured run summary.
- `.omx/research/pareto_analysis_raw_20260513.txt` — raw Pareto enumeration.
- `.omx/state/continual_learning_posterior.json` — updated posterior (25→42 anchors).

CLAUDE.md non-negotiable cross-refs: "Subagent coherence-by-default" (hook #5
mandatory), "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 contest-compliant
hardware" (ship-readiness classification above), "Apples-to-apples evidence
discipline" (CUDA-CPU gap pattern observation kept axis-tagged), Catalog #127 +
#128 (custody validator + locked writes — both fired correctly).
