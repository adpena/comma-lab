---
title: PHASE 4 INTEGRATION — Optimal Stack Predicted Score Band
date: 2026-05-08
author: Subagent SYNTHESIZE (claude-opus-4-7-1m)
status: aggregate-of-byte-anchors prediction; CUDA dispatch required for promotion
predicted_band: [0.139, 0.173] [predicted; aggregate of byte-anchors; CUDA dispatch required for promotion]
predicted_band_council_median: 0.156 [predicted; council-median; CUDA dispatch required for promotion]
score_claim: false
ready_for_exact_eval_dispatch: false
promotion_eligible: false
related_tasks: ["#307 PARADIGM-δεζ Phase 2", "#308 PHASE 4 INTEGRATION"]
focal_anchor: "PR107 apogee CPU [contest-CPU GHA]"
focal_anchor_path: "experiments/results/pr107_apogee_cpu_auth_eval_gha_20260508T124452Z/contest_auth_eval.adjudicated.json"
focal_score_recomputed: 0.1966358879
focal_d_seg: 0.00058931
focal_d_pose: 3.58e-05
focal_archive_bytes: 178392
focal_archive_sha256: "7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb"
---

# PHASE 4 INTEGRATION — Optimal Stack Predicted Score Band

## 1. Purpose and rigor framing

This memo aggregates currently-landed empirical anchors (Phase A0–A6 byte
anchors, A1 infrastructure-blocked, plus existing PR107 component anchors) and
projects a **Phase 4 5-component stacked score** using the closed-form
analytical model in `tac.score_geometry` + `tac.score_geometry_stacking`.

**Per CLAUDE.md "KILL is LAST RESORT": predicted bands cannot be falsified —
only failures of the underlying empirical anchors can drop a Phase 4 lane.**
The band is a planning artifact, NOT a measurement. Promotion requires:

1. Each component anchor lands with [contest-CUDA] evidence at byte-closed runtime, AND
2. The 5-component composed archive runs through `inflate.sh → upstream/evaluate.py`
   on a 1:1 contest-compliant runner and produces a `contest_auth_eval.json`
   on the EXACT archive bytes that were composed.

Until both gates fire, every score in this memo carries
`[predicted; aggregate of byte-anchors; CUDA dispatch required for promotion]`.

## 2. Focal anchor (the baseline we project from)

PR107 apogee — our 2026-05-04 contest entry — has BOTH a CUDA and a CPU
anchor on bit-identical contest hardware. We project from the CPU anchor
because it is the official contest-leaderboard axis (per CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA").

| Field | Value | Source |
|---|---|---|
| Archive bytes | 178,392 | `pr107_apogee_cpu_auth_eval_gha_20260508T124452Z/contest_auth_eval.adjudicated.json` |
| Archive SHA-256 | `7ecb0df1...c29310eb` | same |
| `avg_segnet_dist` (d_seg) | 0.00058931 | same |
| `avg_posenet_dist` (d_pose) | 3.58e-05 | same |
| `compression_rate` | 0.00475136 | same |
| **Recomputed score** | **0.1966358879** | `[contest-CPU GHA Linux x86_64 ubuntu-24.04/20260413.86]` |

CUDA cross-anchor: PR107 apogee `[contest-CUDA T4]` = 0.22933 (gap +0.0327
vs CPU; pose 4.86× wider on CUDA, seg 1.17× wider, rate identical).

## 3. Component delta table (council-derived from Phase A anchors)

Per `.omx/research/grand_council_extreme_rigor_track_1_20260508.md` Decision
1–5 deliberation, the component byte and distortion deltas are:

| Phase | Decision | δ_seg (frac) | δ_pose (frac) | δ_bytes (B) | Anchor source |
|---|---|---:|---:|---:|---|
| A1 | Score-gradient supervision | 10–30% (mid 20%) | 10–25% (mid 17.5%) | 0 | UNANIMOUS HIGHEST PRIORITY; infrastructure-blocked currently |
| A2 | Sensitivity-aware quantization | 0 | 0 | 8,000–15,000 (mid 11,000) | `track1_phase_a2_sensitivity_quant_20260508T154125Z/A2_result.json`; council median |
| A4 | ChARM channel-AR prior | 0 | 0 | 14,000–32,000 (mid 23,000) | A4 toy roundtrip exact (50K params); A4-alt CPU-build 137,531 B at 4.15% rel_err |
| A4-alt (lane_pd_v2) | Pose-deriver / arithmetic-coded poses | 0 | 0 | 3,000–7,000 (mid 5,000) | existing `lane_pd_v2`; council substitution for Decision 4 |
| A5 | Frame-conditional bit budget | 0 | 0 | 1,278 (anchored net byte proxy) | `pr101_frame_conditional_bit_codex_20260508Tlocal/build_manifest.json`; best eta=2.0, latent delta -1,503 B, sidechannel +225 B |

**Per-component caveats:**
- A1 (Decision 2) deltas are predicted from score-gradient closure of the
  proxy-auth gap; verified empirically in OTHER projects (e.g. `feedback_proxy_auth_math_useless`),
  not on PR101/PR107 directly.
- A2 (Decision 3) deltas are byte-only at this Phase A snapshot. The current
  weighted allocation actually REGRESSES vs uniform by 1,933 B at the stub-sensitivity-map
  config; the council range assumes a CERTIFIED sensitivity map will reverse this.
- A4 (Decision 1) toy roundtrip-exact; real-PR101 byte savings are a separate
  unverified gate. Per Quantizr+Carmack: Decision 1 LOWEST priority of the 5.
- A4-alt is the existing `lane_pd_v2`, not a new build; treated as a free bolt-on.
- A5 byte-savings are net of side-info overhead and remain byte-proxy only
  until per-pair score marginals and inflate-runtime schema support land.

## 4. Composed score predictions

### 4.1 Closed-form composition (linear seg/bytes axes; concave pose axis)

Using `tac.score_geometry.contest_score(d_seg, d_pose, archive_bytes)`:

| Scenario | δ_seg | δ_pose | δ_bytes | Composed score |
|---|---:|---:|---:|---:|
| **Conservative (low-end of council range)** | 10% | 10% | 26,278 | **0.17227** [predicted] |
| **Council Median** | 20% | 17.5% | 40,278 | **0.15629** [predicted] |
| **Aggressive (high-end of council range)** | 30% | 25% | 55,278 | **0.13961** [predicted] |

### 4.2 Volterra triple-stacking decomposition (median)

Decomposing A1 (Lane A; seg+pose) × A2 (Lane B; bytes) × A4 (Lane C; bytes)
via `predict_triple_stacking()`:

| Stack | Score | Gain vs focal |
|---|---:|---:|
| Focal (PR107 CPU) | 0.19664 | — |
| A only (A1 score-gradient) | 0.18311 | -0.01352 |
| B only (A2 sensitivity) | 0.18931 | -0.00732 |
| C only (A4 ChARM) | 0.18132 | -0.01531 |
| A + B (no third) | 0.17579 | -0.02085 |
| A + C | 0.16780 | -0.02884 |
| B + C | 0.17400 | -0.02264 |
| **A + B + C (Volterra triple)** | **0.16048** | **-0.03616** |
| + A4-alt 5,000 B | 0.15715 | -0.03949 |
| + A5 1,278 B | 0.15629 | -0.04034 |
| **Final 5-component (median)** | **0.15629** | **-0.04034** |

Stack notes:
- `nominal_stack_ratio = 1.0000` and `triple_pose_correction = 0.0` because
  A2/A4/A4-alt/A5 are pure-byte lanes with zero pose delta. Volterra
  super-additivity applies WHEN multiple lanes touch the pose axis. The pose
  axis is touched only by A1 in this composition; therefore the byte-only
  components stack additively (linear in seg term and rate term, both linear
  axes).
- If A2 or A4 are eventually retrained to also reduce pose distortion (e.g.
  joint-training Decision 2 pose head), the analytical model predicts a
  pose-pose super-additivity bonus per `feedback_volterra_super_additive_pose_stacking_finding_20260507`.
  That bonus is NOT counted here.

### 4.3 Predicted band

Per the conservative/median/aggressive table:

**Predicted band: 0.139–0.173 [predicted; aggregate of byte-anchors; CUDA
dispatch required for promotion]**

**Council median: 0.156 [predicted; CUDA dispatch required for promotion]**

The median places the stack INSIDE the medal cluster (PR101/102/103 = 0.193–
0.195) and ahead of every public PR scored CPU. The aggressive arm of the
band would beat the gold (PR101 = 0.193) by 0.05+ on CPU.

## 5. Sensitivity to the per-component anchors

The composed score is most sensitive to A1 (score-gradient) and A4 (ChARM):

| Risk axis | If FALSIFIED | New median composed |
|---|---|---:|
| A1 (Decision 2) infrastructure-blocked persists, no [contest-CUDA] anchor | drop seg/pose deltas to 0 | 0.16982 |
| A4 (Decision 1) ChARM fails on real PR101 (canonical Ballé already failed at -0.985) | drop A4 23,000 B to 0 | 0.17161 |
| A4 + A1 both fail | drop both | 0.18513 |
| A2 (Decision 3) certified-sensitivity-map STILL regresses vs uniform | drop A2 11,000 B to 0 | 0.16362 |
| All five fail (return to focal) | — | 0.19664 |

Even in the worst case where Decisions 1 and 2 BOTH FAIL, the byte-only
portion (A2 + A4-alt + A5 ≈ 17,278 B saved with no distortion increase, IF
sensitivity map gets certified) lands at ~0.18513 — still beats the focal
PR107 by 0.012, but does NOT enter the medal cluster.

**A1 is the critical-path component.** Its blocker is purely infrastructure
(operator GPU access on Lightning Studio or Vast.ai); the tooling is fully
landed.

## 6. Prerequisite empirical anchors for promotion

Per the predicted-band-cannot-be-falsified rule, the following empirical
anchors MUST land before any Phase 4 lane upgrades from `[predicted]` to
`[contest-CUDA]`:

### Priority 1 — Critical path
1. **A1 [contest-CUDA] anchor** — operator attaches T4 to Lightning Studio
   OR refills Vast.ai → run `tools/dispatch_phase_a1_score_gradient_pr101.py`
   → harvest archive + contest_auth_eval.json → record per-component seg/pose
   delta on real PR101 substrate.
2. **A4 (real PR101) [byte-anchor] anchor** — train ChARM on real
   228,958-element PR101 INT8 weight stream (NOT 50K toy); confirm ≥ 5 KB
   savings vs brotli-q11 baseline at <2% rel_err.
3. **A2 (certified sensitivity) [byte-anchor] anchor** — replace stub
   `sensitivity_map_pr106_20260504_claude/sensitivity_map_stub.pt` with a
   PR101-component-sensitivity map computed against the actual PR101 scorer
   forward path; weighted allocation must beat uniform by ≥ 5 KB at the
   target rms.

### Priority 2 — Deferred
4. **A5 [runtime-cost anchor]** — byte proxy landed at net -1,278 B; verify
   inflate.sh runs ≤ 30 min on T4 with frame-budget side-info path active.
5. **A4-alt [contest-CUDA]** — `cross_paradigm_admm_x_op1_finalizer` archive
   (153,513 B sha 7bbba307) dispatched via `tools/parallel_dispatch_top_k.py`;
   confirm predicted band [0.18, 0.22] holds at score level.

### Priority 3 — Phase 4 closure
6. **5-component composed archive [contest-CUDA AND contest-CPU]** — assemble
   all Phase A winners into one `archive.zip`, run dual-eval per CLAUDE.md
   "Submission auth eval — BOTH CPU AND CUDA"; record both axes in
   `experiments/results/phase4_full_stack_<timestamp>/`.

## 7. Open questions and reactivation criteria

Per CLAUDE.md "KILL is LAST RESORT" — every component below is
DEFERRED-pending-research, not killed. Concrete reactivation criteria:

| Component | Open question | Reactivation criterion |
|---|---|---|
| A1 (score-gradient) | Will gradient quality on a 50-frame random subset transfer to full 600-pair eval? | Single [contest-CUDA] dispatch produces seg or pose improvement ≥ 10% relative. |
| A2 (sensitivity) | Does the certified sensitivity map exist in this codebase, and which forward path computes it? | Certified map produced via `tools/compute_pr101_component_sensitivity.py` (NOT YET WRITTEN); allocator output beats uniform on weighted-vs-uniform delta. |
| A4 (Ballé/ChARM) | Will hyperprior generalize from 50K toy to 228K real PR101? Sister `compressai_balle_FIXED` failed at -0.985. | Real-PR101 byte savings ≥ 5 KB at <2% rel_err. |
| A5 (frame budget) | Does the side-info path keep inflate ≤ 30 min on T4, and do per-pair score marginals justify the redistribution? | Single inflate run captures ≤ 30 min wall-clock with side-info active, and per-pair score marginal evidence is nonnegative. |
| Stack composition | Do byte-only and seg/pose deltas truly compose linearly, or do tensor-count interactions (A2+A4 share allocation budget) erase savings? | Empirical stack archive lands a [contest-CUDA] score that recovers ≥ 80% of the predicted gain. |

## 8. Component anchor staleness audit

For each Phase A row in `.omx/research/phase_a_ablation_anchors_20260508.md`,
this memo's deltas are pinned to:

- A1 deltas (10–30% / 10–25%): council median (no empirical ranging)
- A2 deltas (8–15 KB): council range; current empirical row REGRESSES (-1,933 B at stub map)
- A4 deltas (14–32 KB): council range; only A4 toy + A4-alt CPU-build land at this snapshot
- A4-alt 3–7 KB: existing lane_pd_v2 anchor
- A5 1,278 B: `tools/pr101_frame_conditional_bit_anchor.py` local byte proxy

If new Phase A anchors land that contradict the council range, this memo's
predicted band MUST be re-derived. The cathedral autopilot recommender
(`tools/cathedral_autopilot.py`) is the canonical consumer; a regenerated row
would invalidate the score columns above.

## 9. Cross-references

- `.omx/research/grand_council_extreme_rigor_track_1_20260508.md` — council deliberation
- `.omx/research/track_1_co_designed_substrate_design_20260508_claude.md` — Track 1 substrate scope
- `.omx/research/phase4_optimal_stack_design_20260508_claude.md` — Phase 4 stack design (this memo's quantitative companion)
- `.omx/research/phase_a_ablation_anchors_20260508.md` — Phase A anchor table
- `docs/paper/phase4_paper_harness_blueprint_20260508.md` — paper-harness empirical TODO
- `experiments/results/pr107_apogee_cpu_auth_eval_gha_20260508T124452Z/contest_auth_eval.adjudicated.json` — focal anchor
- `src/tac/score_geometry.py` + `src/tac/score_geometry_stacking.py` — analytical model
- `feedback_volterra_super_additive_pose_stacking_finding_20260507.md` — pose-pose Volterra closed form
- `feedback_dual_cpu_cuda_auth_eval_mandatory_20260508.md` — dual-eval mandate
