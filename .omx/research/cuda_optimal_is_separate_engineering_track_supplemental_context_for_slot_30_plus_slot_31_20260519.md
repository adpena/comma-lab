# Supplemental context — CUDA-optimal is a SEPARATE engineering track, not a closure of CPU-CUDA gap

**Date:** 2026-05-19 (UTC)
**Authority:** Operator verbatim 2026-05-19 correction
**For consumption by:** SLOT 30 (`afceb9c62403f9781` — CPU-vs-CUDA + MPS deep analysis) + SLOT 31 (`a5f6a29d5e7c774fc` — prior context location + HF Jobs fix) + any future operator-routed follow-up on the device-class optimization space

## Operator verbatim quote

> "the optimal solution for cuda requires different engineering and techniques or addtional work on top of the optimal solution for cpu but we haven't fully explored or udnerstood that space or the frontier yet"
>
> — Operator, 2026-05-19

## What this overrides

My earlier framing in slot 30's dispatch + my supplemental insights treated CPU-vs-CUDA mismatch as "engineer the CUDA disadvantage OUT" — i.e., close the gap so CUDA matches CPU on the same archive. The operator's correction reframes this as:

**CUDA-optimal is its own engineering target**, NOT a closure of the CPU-CUDA gap on the same archive. Specifically:
- The CPU-optimal solution may require techniques X, Y, Z
- The CUDA-optimal solution may require DIFFERENT techniques A, B, C — or X+Y+Z+A+B+C (additional work on top of CPU-optimal)
- We haven't fully explored OR understood the CUDA-optimal space / frontier yet

## Concrete implications

### Implication 1: Two separate frontier-pursuit threads, not one

Our actual canonical frontier pointer per slot 14 records TWO DIFFERENT archives:
- **CPU-axis frontier**: `0.1920513169 / archive 6bae0201fb08 / lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean` (pr101 fec6 family)
- **CUDA-axis frontier**: `0.2053300290 / archive 9cb989cef519 / lane pr106_format0d_latent_score_table` (pr106 family)

These are DIFFERENT engineering solutions. pr101_fec6 was optimized for the CPU axis (and incidentally has whatever CUDA score it has — currently unknown until slot 29's paired auth eval lands). pr106_format0d was optimized for the CUDA axis.

The CUDA-optimal-engineering-space remains LARGELY UNEXPLORED per operator. We don't know what techniques would specifically optimize for CUDA-axis scoring beyond pr106_format0d's current 0.20533.

### Implication 2: The per-archive CUDA-vs-CPU drift IS REAL but is a SECONDARY phenomenon

PR102 +0.033 CUDA penalty on the SAME archive IS a real measurement-drift phenomenon. Slot 16's MPS engineering corrections (Kahan / pinned softmax / fp32 matmul) sister-apply to closing this PER-ARCHIVE drift.

But that's DIFFERENT from the OPERATOR'S concern: the CUDA-optimal-engineering-space exploration. Closing per-archive drift makes CUDA EVAL more reliable; exploring CUDA-optimal techniques is about CONSTRUCTING ARCHIVES that specifically maximize CUDA-axis score.

### Implication 3: Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" we need both

The contest leaderboard ranks by CPU, so CPU-axis is primary score-lowering target. But CUDA-axis is published alongside; explored CUDA-axis optimization is HIGH-VALUE for:
- Paper completeness (showing the full Pareto frontier across deployment targets)
- Production/openpilot relevance (CUDA may be canonical for some edge deployment classes)
- Cathedral autopilot routing (some substrates may be CUDA-axis-superior; informs dispatch decisions)
- Defensive: if contest changes to CUDA-axis ranking in a future round, we're not blindsided

### Implication 4: The canonical equations registry (slot 19) should support per-device-class equation families

Slot 19's `tac.canonical_equations.CanonicalEquation` dataclass currently encodes generic equations. The slot 30 candidate equation `eq_cross_device_drift_by_kernel_class_v1` should be DECOMPOSED into:
- `eq_cpu_optimal_archive_techniques_v1` — techniques + architectures + training curricula that maximize CPU-axis score
- `eq_cuda_optimal_archive_techniques_v1` — techniques + architectures + training curricula that maximize CUDA-axis score (LARGELY UNEXPLORED PER OPERATOR)
- `eq_per_archive_cross_device_drift_v1` — the per-archive measurement-drift phenomenon when scoring an archive on a different device than it was optimized for
- `eq_pareto_cpu_cuda_frontier_v1` — the joint Pareto frontier across both axes; do dominated points exist? Are there "CPU-only" / "CUDA-only" / "Pareto-optimal" archive classes?

### Implication 5: The findings Lagrangian (slot 24) should support per-device-class objective functions

Slot 24's `tac.findings_lagrangian.LagrangianObjective` should accept a `target_device_class` parameter so:
- `target_device_class="contest_cpu"` → optimizes for CPU-axis score
- `target_device_class="contest_cuda"` → optimizes for CUDA-axis score
- `target_device_class="pareto_optimal"` → optimizes for joint frontier
- `target_device_class="agnostic"` → optimizes for axis-independent properties

The active-inference action selector should also support per-device-class targeting — recommending experiments that maximize info gain ABOUT the CUDA-optimal space (which is currently under-explored) vs the CPU-optimal space.

## For slot 30's CPU-vs-CUDA + MPS deep analysis

REFRAME the dispatch's primary question from:
- ~~"can we engineer the CUDA disadvantage out"~~

To:
- **"what does the CUDA-optimal engineering space look like, and what's the relationship to the CPU-optimal engineering space?"**

Specifically, the analysis should cover:
1. **CUDA-axis-specific engineering techniques** — what would techniques X, Y, Z (CPU-optimal) plus A, B, C (CUDA-specific) look like? Sister exploration of CUDA-friendly architecture (TF32-tolerant pose-head numerics / DALI-friendly kernel layout / cuDNN-canonical convolutions / etc.)
2. **Pareto frontier characterization** — across all our archives, what does the (CPU-score, CUDA-score) joint distribution look like? Are there Pareto-optimal points? Pareto-dominated points?
3. **Engineering directions FOR CUDA-optimal exploration** — concrete operator-routable proposals (e.g., "train substrate explicitly with TF32 emulated in training loop so the CUDA forward path matches training-time numerics" / "use CUDA-aware DALI pipeline as canonical training data path" / etc.)
4. **MPS-vs-CUDA-vs-CPU triple-axis exploration**: MPS local training as CPU-axis-canonical-or-CUDA-axis-canonical research signal — slot 16's findings already establish MPS local is CPU-side bit-identical for some architectures (PR107 6e-6 between M5 Max macOS-CPU and GHA Linux x86_64)

## For slot 31's prior context location

ALSO search for prior CUDA-optimal-specific analysis. The framing "the CUDA-optimal space is largely unexplored" implies SOME prior analysis exists but is incomplete. Likely candidates:
- Prior CUDA-axis dispatch results (cathedral autopilot anchors on CUDA axis)
- pr106 family's CUDA-axis optimization journey
- Any prior writeup/draft on CUDA-axis-specific techniques

Include these in the synthesis memo for slot 30 consumption.

## For the operator check-in package (slot 29)

The PR body should ACKNOWLEDGE this asymmetry honestly:
- Our CPU-axis frontier: pr101_fec6 at 0.19205 (CPU-optimal engineering)
- Our CUDA-axis frontier: pr106_format0d at 0.20533 (CUDA-optimal-known; CUDA-optimal-frontier-unexplored)
- Submission decision: per contest leaderboard ranks by CPU, submit pr101_fec6
- Reporting: both axes published; the CUDA score reflects pr101_fec6's CUDA performance (not pr106's), since you can only submit ONE archive

The PR body should NOT claim CUDA-axis-optimality on pr101_fec6 if that's not true. Be honest about the asymmetry.

## Cross-references

- Slot 14 canonical frontier pointer (per-axis-per-archive separation captured correctly)
- Slot 19 canonical equations registry (candidate equation decomposition per device class)
- Slot 24 findings Lagrangian PARALLEL DUAL-TRACK build (per-device-class objective function support)
- Slot 29 PR pre-submission canonicalization (paired auth eval per archive per axis)
- Slot 30 CPU-vs-CUDA + MPS deep analysis (PRIMARY consumer of this supplemental context)
- Slot 31 prior CUDA-specific analysis location (SECONDARY consumer; integrates prior CUDA-axis work)
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable
- CLAUDE.md "Contest vs production target modes — non-negotiable" (canonical 4-mode taxonomy: contest_one_video_replay / contest_generalized / production_generalized / production_edge_adaptive — each has different optimal device-class)

— Claude-main 2026-05-19 (supplemental context for slot 30 + slot 31 + future operator-routed follow-up)
