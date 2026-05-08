# Track 1 Phase A4 — recursive adversarial review log

**Date:** 2026-05-08
**Subject:** `experiments/train_charm_50k_toy_substrate.py` (ChARM 50K toy substrate trainer)
**Reviewer:** fork worker (parent: claude:main, council subagent: aa63e5fc)
**Council scope:** parent already ran 6 rounds on the broader Track 1 design (commit 77cbb37a). This log covers the FILE-LEVEL recursive review per CLAUDE.md non-negotiable: 3 consecutive clean passes required.

## Round 1 — math + dead code lens (5 findings, all CRITICAL or MEDIUM)

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | CRITICAL | Differential entropy formula `0.5·log2(2π σ²)` was missing the `e` factor — should be `0.5·log2(2π·e·σ²)`. Conflated nat-units in cross-entropy correction with bit-units. | Rewrote rate computation to split diff_entropy_bits + correction_bits with explicit `log2_e = 1.4426950408889634` conversion. Verified rate now in correct bit units. |
| 2 | MEDIUM | `torch.frombuffer(buf.read(n), dtype=torch.int8).clone()` raises a non-writable buffer warning that pollutes stdout on every decode. | Switched to `torch.from_numpy(np.frombuffer(...).copy())`. Warning eliminated. |
| 3 | LOW | Dead `buf.write(b"" if False else b"")` line in `encode_weight_with_charm`. | Removed. |
| 4 | LOW | Scale serialized as float64 (8 bytes) in archive blob; INT8 weights only need fp32 scale precision. | Switched to `struct.pack(">f", scale)` (4 bytes). Saves ~4 B per tensor × ~30 tensors. |
| 5 | LOW | `import struct` was inside the function body — moved to header for clarity. | Hoisted alongside other imports. |

Smoke after R1 fixes: PASS — rate finite, decreasing 2.45M→2.36M B over 5 epochs, roundtrip exact, model params 46,995.

## Round 2 — engineering correctness lens (3 findings, all CRITICAL or MEDIUM)

| # | Severity | Finding | Fix |
|---|---|---|---|
| 6 | MEDIUM | `from compressai.entropy_models import EntropyBottleneck, GaussianConditional` imported but never used. The toy implements rate from scratch. | Replaced with `import compressai` presence check. Production Phase C variant should wire `GaussianConditional`; that's documented in the comment. |
| 7 | CRITICAL | `lambda_R_target=1e-4` made the rate-loss term ~340× the recon term at start. Model would collapse to all-zero weights (low rate, terrible recon). | Rebalanced to `1e-6` (rate contribution ~3.4 vs recon 0.5 → moderate pressure). Verified: smoke loss after R2 is 1.59 vs 109 before. |
| 8 | MEDIUM | `build_archive` reported `total_compressed_bytes` but didn't distinguish the actual naive INT8 dump bytes from the [predicted] hyperprior-achievable rate. Misleading for Phase C planning. | Split manifest into `naive_int8_total_bytes` (actual zip contents) + `ideal_hyperprior_rate_bytes_predicted` (the rate-loss lower bound, achievable with `GaussianConditional` + range coding in Phase C). |

Smoke after R2 fixes: PASS — loss balance correct, both metrics tracked, roundtrip exact.

## Round 3 — CLAUDE.md schema + edge case lens (4 findings, all LOW or MEDIUM)

| # | Severity | Finding | Fix |
|---|---|---|---|
| 9 | LOW | `import numpy as np` was inside `decode_weight_with_charm`. Hoist to top-level. | Moved. |
| 10 | LOW | INT8 range was `[-127, 127]` — wasting `-128`. Full INT8 is `[-128, 127]`. | Updated `clamp(min=-128.0, max=127.0)`. |
| 11 | MEDIUM | argparse `--device` accepted `mps` even though `train_loop` raises on it. Cleaner to fail at argparse. | Removed `mps` from choices. |
| 12 | MEDIUM | `torch.load(ckpt_path, weights_only=False)` not annotated for preflight Check 14 (`preflight_loader_format_safety`). | Added `WEIGHTS_ONLY_FALSE_OK:` comment with rationale (locally-produced ckpt). |

Smoke after R3 fixes: PASS — all checks green.

## Round 4 — CLEAN PASS

Re-reviewed the full file with fresh eyes:
- Math: differential entropy formula + cross-entropy correction proper, log2_e correctly applied ✓
- Engineering: numpy at top, INT8 range full, mps removed from argparse, weights_only annotated ✓
- Schema: manifest has all required CLAUDE.md fields (`evidence_grade`, `score_claim=False`, `ready_for_exact_eval_dispatch=False`, naive vs ideal byte split, falsification_criteria, council_finding_reference, tier_0_reference) ✓
- Lane registered at L0 (`tools/lane_maturity.py add-lane track1_phase_a4_charm_50k_toy --phase 4`) ✓
- Smoke passes (rate finite, decreasing, roundtrip exact, 47K params) ✓
- No /tmp paths ✓
- EMA properly snapshot+restore at eval (`finally:` block restores live weights) ✓
- eval_roundtrip threaded with `simulate_eval_roundtrip` (384→874→uint8→384) ✓
- noise_std=0.5 (Hotz fix) propagated ✓
- parametrize_strip applied via `strip_parametrize_hooks(state, drop_internal=True)` before archive build ✓
- CUDA-required default with explicit `--device cpu` opt-in + banner ✓
- compressai presence check (warns clearly if missing) ✓

**No findings. CLEAN.**

## Round 5 — CLEAN PASS

Adversarial composition lens (does this conflict with parent council subagent's design?):
- Architecture (TinyHNeRVToy50K): matches Quantizr-position (DSConv + FiLM, ~50K params target) ✓
- Hyperprior (CharmHyperprior): ChARM 2020 channel-conditional spec per Ballé council position ✓
- Training (eval_roundtrip + EMA + lambda_R warmup): matches Dykstra/Boyd recommendations ✓
- Evidence semantics (`evidence_grade=contest_cuda_pending`, `score_claim=False`): matches CLAUDE.md gates B1/B2 ✓
- Lane id (`track1_phase_a4_charm_50k_toy`): matches dispatch wrapper expectation ✓
- Falsification criteria (compressed < 30 KB at <5% recon): matches council memo ✓

**No findings. CLEAN.**

## Round 6 — CLEAN PASS

Adversarial deployment lens (will this dispatch and harvest correctly?):
- `experiments/results/track1_a4_charm_50k_toy_<TS>/` directory layout: archive.zip, archive_sha256, build_manifest.json, provenance.json, checkpoint.pt — matches harvest expectations ✓
- Build manifest includes `archive_zip_bytes`, `archive_sha256`, `total_compressed_bytes`, `naive_int8_total_bytes`, `ideal_hyperprior_rate_bytes_predicted`, `falsification_criteria` ✓
- Provenance includes `git_commit` (via `_git_head_short`), `torch_version`, `compressai_available`, `device`, `seed`, `council_memo_commits` ✓
- CLI supports `--build-archive-only` for re-build without retrain ✓
- Smoke writes both stdout AND `smoke_log.json` for pipeline integration ✓
- No `/tmp` paths anywhere in artifacts (all under `experiments/results/`) ✓

**No findings. CLEAN.**

## Verdict

**3 consecutive clean passes (R4 / R5 / R6) achieved per CLAUDE.md non-negotiable.** Counter satisfied. File is dispatch-ready pending operator authorization for the actual Lightning T4 spend ($15, ~6h, lane `track1_phase_a4_charm_50k_toy` at L0 awaiting gate progression).

## Next steps (handoff)

1. Operator green-lights Lightning T4 dispatch
2. Parent agent (or operator) invokes `tools/dispatch_phase_a_track_1_ablations.py --decision A4 --substrate toy_50k`
3. Dispatch wrapper claims lane via `tools/claim_lane_dispatch.py`, runs the full 500-epoch training on T4, harvests artifacts to `experiments/results/track1_a4_charm_50k_toy_<TS>/`
4. Harvest validates: `total_compressed_bytes < 30000` AND `eval_l_recon < 0.05` → A4 PASSES, gate G5 transitions RED → GREEN, Phase C unblocked
5. If A4 FAILS: per CLAUDE.md "KILL is LAST RESORT", retire measured config; reactivation requires (a) deeper hyperprior architecture (ChARM with proper masked AR), (b) score-gradient supervision (Decision 2) added to loss, (c) larger toy substrate (~100K params)

## Cross-references

- `.omx/research/grand_council_extreme_rigor_track_1_20260508.md` — parent council 22-member deliberation
- `.omx/research/track_1_co_designed_substrate_design_20260508_claude.md` — original Track 1 design
- `.omx/research/grand_council_track_1_EV_update_post_tier_0_20260508.md` — Tier 0 finding (2,228 B headroom)
- `tools/mdl_lower_bound_calculator.py` — Phase A0 deliverable (already landed at 4373edb2)
- `tools/dispatch_phase_a_track_1_ablations.py` — Phase A actuator (landed at 79f829b4)
- Council memo R6 commit: `77cbb37a` (parent's 6-round design review)
- This file: `track1_a4_recursive_review_20260508.md` (file-level review, 6 rounds)
