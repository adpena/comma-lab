# Contest submission: CPU-axis frontier 0.19205 — PR101-grammar HNeRV with FEC6 frame-conditional K=16 selector

**Primary CPU evidence:** `0.1920513169` `[Modal Linux x86_64 CPU reproduction; contest/GHA host validation pending]`
**Paired CUDA-axis evidence (same archive bytes):** `0.2262100217` `[Modal T4 CUDA replay]`
**Archive SHA-256:** `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
**Archive size:** `178,517` bytes
**Inflate runtime tree SHA-256:** `f67b5b52ca1f11e1a582c53965d88ef738bef86d425b82abdf2e98f3f3fd9166` (CPU run) / `12d4315dcbf0943f07fcd357eaf06b126a999c252f8edeb2681179831248df04` (CUDA run; same content SHA `6811f28c2116757851b4a6e68a5bdefd7866b4da1867eb13b3c62405de8834df`)

Lower score is better.

## 1. Claim

Per the contest scoring formula `S = 100·d_seg + √(10·d_pose) + 25·(archive_bytes / 37,545,489)` (`upstream/evaluate.py:92`):

| axis | seg dist | pose dist | archive bytes | score | runner |
|---|---:|---:|---:|---:|---|
| `[Modal Linux x86_64 CPU; contest/GHA host validation pending]` | `0.00056029` | `0.00002943` | `178,517` | **`0.1920513169`** | Modal CPU (linux x86_64); host-bot validation pending |
| `[Modal T4 CUDA replay]` | `0.00066299` | `0.00016846` | `178,517` | `0.2262100217` | Modal T4 |

Both rows derived from the same archive bytes (verified SHA-256 match).

Evidence artifacts:

```
experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/contest_auth_eval.json
experiments/results/modal_auth_eval/archive_6bae0201fb08/contest_auth_eval.json
experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip
experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/inflate.sh
```

The CPU number is a Modal Linux x86_64 CPU reproduction, not a host-bot/GitHub Actions validation. We keep that distinction explicit until a same-axis host-bot artifact exists.

## 2. What changed vs PR101

This packet starts from the public PR101 HNeRV-family grammar and adds FEC6: a `K=16` frame-conditional per-pair mode selector encoded with a compact fixed-Huffman selector stream. The selector is chosen offline for the fixed 600-pair contest video and consumed deterministically by `inflate.py`; no scorer or search runs at inflate time.

Inherited from PR101 GOLD (Jimmy / "Quantizr"):
- HNeRV decoder architecture (~88K params, FiLM-conditioned depthwise-separable CNN, ~64KB FP4)
- FP4 asymmetric codebook `[0, 0.5, 1, 1.5, 2, 3, 4, 6]` with sign nibble + block-wise fp16 scales
- `qpose14 + qzs3` pose-encoding wire format
- The "encode only frame-0 masks; warp frame-1" insight

Novel in this submission (FEC6 = Frame Exploit Compactor v6):
1. **K=16 frame-conditional per-pair mode palette** (vs PR101's K=8 static modes). Modes include `none`, `frame0_blue_chroma_amp_1`, `frame0_red_chroma_amp_1`, `frame0_blue_tile_*`, `frame0_chroma_offset_*`.
2. **Fixed-Huffman codebook on selector indices** (vs raw-byte storage in PR101 GOLD). 4-bit naïve cost (4 × 600 = 300 bytes) is compacted to roughly 107 bytes for the FEC6 stream via the fixed-Huffman codebook designed against the empirical selector-mode distribution observed on `upstream/videos/0.mkv`.
3. **Per-pair selector decision is offline.** Selector indices are precomputed against the SegNet/PoseNet response surface during candidate enumeration; the inflate path is fully deterministic — no on-device search.
4. **Runtime contract:** archive bytes are byte-stable and inflate path is byte-stable for a fixed hardware axis.

## 3. Reproduce

One pinned path that exists in the current checkout:

```bash
# 1. Verify archive bytes match the canonical SHA-256.
sha256sum experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip
# expected:
# 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf

# 2. Run the contest auth-eval pipeline on Linux x86_64 with CPU.
python experiments/contest_auth_eval.py \
  --archive experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip \
  --inflate-sh experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/inflate.sh \
  --upstream-dir upstream \
  --video-names-file upstream/public_test_video_names.txt \
  --device cpu \
  --expected-runtime-tree-sha256 f67b5b52ca1f11e1a582c53965d88ef738bef86d425b82abdf2e98f3f3fd9166
```

For a CUDA replay (paired context, not the primary claim) substitute `--device cuda` and the CUDA runtime-tree SHA `12d4315dcbf0943f07fcd357eaf06b126a999c252f8edeb2681179831248df04`.

The exact Modal CPU replay command used to produce the headline number is in the canonical CPU auth-eval JSON (`experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/contest_auth_eval.json`).

## 4. Reproducibility verification

We re-verified the headline numbers from canonical posterior state and paired artifacts on 2026-05-19:

| axis | canonical pointer score | paired auth-eval JSON score | drift | hardware |
|---|---:|---:|---:|---|
| `[contest-CPU]` | `0.1920513169` | `0.1920513169` | `0` (exact match) | `linux_x86_64_cpu` (Modal) |
| `[contest-CUDA]` | `0.2262100217` | `0.2262100217` | `0` (exact match) | `linux_x86_64_t4` (Modal) |

Sources:
- `.omx/state/continual_learning_posterior.json` — canonical frontier posterior
- `experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/contest_auth_eval.json` — paired CPU
- `experiments/results/modal_auth_eval/archive_6bae0201fb08/contest_auth_eval.json` — paired CUDA

The CPU and CUDA score components decompose identically on rate (`25·R = 0.118867`); the score split between axes is driven entirely by `d_seg` and `d_pose` distortions, consistent with bit-identical archive bytes flowing through device-dependent floating-point paths in upstream `evaluate.py`.

## 5. Limitations

- **Modal CPU is not yet host-bot/GHA-validated for this exact archive.** The contest's automated `[contest-CPU]` leaderboard score is produced by the maintainer's CI runner; we have not yet seen a host-bot comment validating this archive at the headline number. Treat the headline number as a Modal Linux CPU reproduction until a same-axis host-bot artifact exists.
- **The Modal T4 CUDA score is paired context, not the promoted axis.** This packet is CPU-axis first because that is where it has its strongest evidence and where the public-leaderboard comparison is most direct.
- **CPU/CUDA score split is observed and documented, not causally attributed.** The same archive bytes score differently on CPU vs CUDA paths; we have paired output hashes that prove the device-axis split is real, but precise causal attribution to specific floating-point kernels remains hypothesis-labeled until controlled toggle experiments isolate each factor.
- **This packet is contest-specific.** It uses offline access to the fixed 600-pair contest video for selector precomputation; it is not framed as directly production-deployable.

## 6. Acknowledgements + appendix links

This work builds directly on PR101 (Jimmy / "Quantizr") and incorporates the composable selector-axis pattern from PR103 (rem2 silver). It would not exist without the open submissions, the contest scoring infrastructure, and the maintainer review work that anchors the public leaderboard.

Long research dossier (internal — useful context, not part of the submission body):

```
docs/pr_writeups/cpu_frontier_fec6_20260517.md
```

Senior-engineer/taste review of an earlier draft of this body (codex):

```
.omx/research/pr_body_senior_engineering_taste_review_20260517_codex.md
```

Pre-submission-surface adversarial review (codex):

```
.omx/research/fec6_cpu_frontier_submission_surface_adversarial_review_20260517_codex.md
```

Happy to discuss engineering details or production applicability with the comma.ai team.
