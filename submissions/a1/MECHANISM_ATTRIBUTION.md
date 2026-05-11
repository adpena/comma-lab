# A1 mechanism attribution table (entry packet — D5 expansion #2)

What A1 specifically does, mechanism by mechanism, with verification paths.
This memo is the disambiguator-of-attribution for the 5-turn council review.

## Pedigree

A1 is `track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6` —
a score-gradient-trained finetune of the PR101 HNeRV architecture with
inflate-time bias-correction sidecar.

## Mechanism rows

| Mechanism | Description | Where in this packet | Cross-ref |
|---|---|---|---|
| **HNeRV-cluster representation** | renderer (HNeRV decoder, latent_dim=28, base_channels=36) + per-frame-pair latent (N_PAIRS=600 × 28-dim) + decoder Brotli-split packing (PR101 grammar) | `src/model.py` (HNeRVDecoder class); `src/codec.py` (decode_decoder_compact + decode_latents_compact) | PR101 HNeRV public source; PR101 grammar paper-trail in `feedback_pr101_*_landed_*.md` |
| **Score-gradient training** | trainer differentiated through SegNet + PoseNet via `eval_roundtrip=True` + differentiable `rgb_to_yuv6` patch + score-domain Lagrangian (α·B/N + β·d_seg + γ·√(10·d_pose)) on `upstream/videos/0.mkv` | training logs at `experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal/harvested_artifacts/logs/`; trainer source `experiments/train_score_gradient_pr101_finetune.py` | CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE"; `feedback_codex_finding_pr101_synthetic_targets_FIXED_20260508.md` |
| **Inflate-time bias-correction sidecar** | `apply_latent_sidecar()` at inflate-time applies per-frame-component bias correction (subtracts 1.0 from R+B of frame0 and G of frame1, computed from PR101 train-vs-eval discrepancy probe) | `src/codec.py::apply_latent_sidecar()`; `inflate.py` lines 78-80 (`up[:, 0, 0].sub_(1.0); up[:, 0, 2].sub_(1.0); up[:, 1, 1].sub_(1.0)`) | `feedback_a1_inflate_time_bias_correction_sweep_landed_20260509.md` — V7 (pr101_stack_pr102_red) variant; bias-magnitude sweep across V0-V10 |
| **Latent alignment ("latentalign")** | latent permutation/alignment fix from the original PR101 finetune that ensured per-frame latents track ground-truth pair ordering | training memo + import-path-fix in `experiments/train_score_gradient_pr101_finetune.py` | `feedback_a1_dispatch_recursive_review_20260508.md` import-path-fix discussion |
| **Archive grammar (PR101 monolithic single-file)** | single ZIP member `x` containing: `uint32 LE D` header + `(D-4)` bytes encoded decoder blob (PR101 split-Brotli) + `15387` bytes latent_blob + remaining bytes sidecar_blob | `inflate.py::parse_a1_finetuned_archive()`; archive layout offsets fixed in `LATENT_BLOB_LEN = 15387` constant in `src/codec.py` | CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lesson 3 (monolithic single-file `0.bin`-or-`x` archive grammar) — A1 uses single member named `x` |
| **Inflate runtime (≤100 LOC)** | inflate.sh 599 bytes (`set -euo pipefail`, while-read pattern, 3 positional args); inflate.py 3,333 bytes (CUDA-or-CPU autoselect, no MPS fallback per CLAUDE.md, no scorer load at inflate-time) | `inflate.sh` + `inflate.py` in this dir | CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lesson 4 (≤100 LOC inflate budget); CLAUDE.md "Strict scorer rule" — inflate does NOT load PoseNet/SegNet |
| **No-MPS guard** | `inflate.py` line 56: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` — explicit no-MPS fallback ternary | `inflate.py:56` | CLAUDE.md FORBIDDEN_PATTERNS (`mps-fallback-trap`); A1 inflate is structurally MPS-safe by construction |
| **Dual-eval custody (both axes)** | CPU eval on GHA Linux x86_64 ubuntu-24.04 → 0.19285 ; CUDA eval on Modal Tesla T4 → 0.22635 ; both on EXACT same archive bytes (sha256 verified) | `contest_auth_eval.cpu.json` + `contest_auth_eval.cuda.json` + `dual_eval_adjudicated.json` | CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"; `feedback_a1_dual_cuda_dispatch_landed_20260509.md` |

## Validation paths

Each mechanism has an associated empirical or council-grade validation:

| Mechanism | Validation grade | Anchor |
|---|---|---|
| HNeRV-cluster representation | `[contest-CUDA]` + `[contest-CPU]` on archive sha `87ec7ca5...492b5` | `submissions/a1/contest_auth_eval.{cuda,cpu}.json` |
| Score-gradient training | `[contest-CUDA]` and `[contest-CPU]` (mechanism delivers the dual-anchor score) | `feedback_a1_dual_cuda_dispatch_landed_20260509.md` |
| Inflate-time bias-correction | `[contest-CPU GHA Linux x86_64]` sweep V0-V10 (V7 selected as A1's bias pattern) | `feedback_a1_inflate_time_bias_correction_sweep_landed_20260509.md` |
| Latent alignment | empirical (lr2e6 retrain after `importpathfix` produced the dual-anchor result) | dispatch ledger row `lane_a1_dual_cuda_dispatch_20260509` |
| Archive grammar | byte-level structural validation (`pre_submission_compliance_check.py --contest-final --strict` stub) | `pre_submission_compliance.contest_final.json` (stub committed; final result requires operator-trigger) |
| Inflate runtime | structural: `inflate.sh` ≤100 LOC, `inflate.py` <200 LOC, no scorer imports | `inflate.sh` + `inflate.py` in this dir; CLAUDE.md HNeRV parity lesson 4 |
| No-MPS guard | grep on `inflate.py:56` shows no `mps` substring | direct file inspection |
| Dual-eval custody | both axes 1:1 contest-compliant; SHA verification on archive bytes | `dual_eval_adjudicated.json` lines for `archive_sha256` |

## What A1 does NOT do (negative claims for council review)

- A1 does NOT load PoseNet or SegNet at inflate time. (Strict-scorer-rule respected.)
- A1 does NOT bundle scorer weights in the archive. (~73MB scorer would destroy the rate term.)
- A1 does NOT depend on MPS. (Explicit cuda-or-cpu ternary.)
- A1 does NOT use `/tmp` paths in any persisted artifact in this submission. (Per CLAUDE.md FORBIDDEN_PATTERNS the eval JSONs reference Modal-side `/tmp/modal_auth_eval/` paths as forensic scratch only; submission packet README does NOT cite these as durable evidence.)
- A1 does NOT use synthetic data in non-smoke training. (Trainer fix landed 2026-05-08 per `feedback_codex_finding_pr101_synthetic_targets_FIXED_20260508.md`.)
- A1 does NOT make any score claim outside of the dual-eval custody anchors recorded here.

## Cross-references

- `feedback_a1_dual_cuda_dispatch_landed_20260509.md`
- `feedback_a1_inflate_time_bias_correction_sweep_landed_20260509.md`
- `feedback_codex_finding_pr101_synthetic_targets_FIXED_20260508.md`
- `feedback_grand_council_5_design_decisions_review_20260511.md` Decision 5 mechanism-attribution-table expansion
- CLAUDE.md "Strict scorer rule" + "HNeRV parity discipline" + "Submission auth eval"
