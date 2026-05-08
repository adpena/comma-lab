# Phase 4 Paper Harness Blueprint

**Date**: 2026-05-08
**Author**: Subagent INTEGRATE
**Status**: blueprint — requires Phase 2-4 empirical anchors before populating
**Related task**: #308 PHASE 4 INTEGRATION

## 1. Purpose

Specify the empirical comparison harness needed for paper-ready evidence vs
the public PR101/PR103/PR106 frontier (0.193/0.195/0.195 medal band). Until
Phase 2-4 dispatches land, this is a structured TODO — the row-types and
rigor requirements must be in place before any score row enters the paper.

## 2. What "paper-ready" means

A paper-ready row has ALL of:

1. **Exact archive bytes**: SHA-256 + size in bytes from the ZIP that produced
   the score
2. **Exact CUDA auth-eval JSON**: `contest_auth_eval.json` with seg_dist,
   pose_dist, rate term, total score, runtime tree custody
3. **Sample count**: `n_pairs = 600` (non-overlapping, matches upstream
   `evaluate.py`)
4. **Component breakdown**: archive member sizes (renderer.bin, masks.*, poses.*)
5. **Reproducibility recipe**: command line, git hash, profile dict
6. **Compliance status**: scorer-at-inflate check, deterministic-ZIP check,
   strict-scorer-rule attestation

A row tagged `[contest-CUDA]` without all of the above is NOT paper-ready;
re-tag as `[predicted-band]` or `[CPU-prep, planning-only]`.

## 3. Comparison frontier

| System | Score | Archive bytes | Notes |
|---|---:|---:|---|
| PR #101 (SajayR — gold) | 0.193 | ~158 KB | Public (4h race window 2026-05-04) |
| PR #103 (rem2 — silver) | 0.195 | ~155 KB | 241 LOC bolt-on; AC encoder |
| PR #102 (EthanYangTW — bronze) | 0.195 | ~154 KB | inference-time scale tuning |
| PR #100 (BradyMeighan — substrate) | 0.1954 | ~157 KB | hnerv_lc_v2; medal-PR substrate |
| PR #106 (`belt_and_suspenders`) | 0.20946 | 186,080 | Local A++ HNeRV rate anchor (predecessor) |
| PR103-on-PR106 (active) | 0.20898 | 185,578 | Active local A++ rate anchor 2026-05-08 |
| PR #107 (`apogee` — our 2026-05-04 entry) | 0.229 | ~196 KB | ~11th place; race-window failure mode |
| Quantizr (ref) | 0.33 | 299,970 | Architecture lane reference (Selfcomp/Quantizr-class) |

Per `feedback_top3_PRs_are_boltons_on_PR100_substrate_20260507`: medal PRs
all derive from the PR #100 substrate. Engineering velocity > novel theory at
this score band.

## 4. Required harness components

### 4.1 Archive pinning + custody (existing)

- `submission_archive.require_valid_archive()` (already canonical)
- `tools/archive_size_audit.py` (existing)
- `scripts/pre_submission_compliance_check.py --contest-final --strict`
  with `--expected-archive-sha256` + `--expected-archive-size-bytes`
- New: `tools/build_paper_comparison_table.py` — emits the comparison row
  with archive SHA-256, byte breakdown, scorer JSON cross-reference. NOT
  YET WRITTEN.

### 4.2 Scorer-distortion attribution

Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent":
at PR106 frontier (pose_avg ~3.4e-5), POSE marginal sensitivity is 2.71×
SegNet's. The paper harness must report:

- `seg_dist` (raw float, sample mean)
- `sqrt(10 · pose_dist)` (the contest-scoring transform)
- `25 · archive_bytes / 37,545,489` (the rate term)
- Marginal value-per-byte at the row's operating point
- If the row trades pose-for-seg (PR97 anti-pattern), explicit caveat

### 4.3 Component ablation rows

For each of δ, ε, ζ, the paper §6 needs:

| Row | δ | ε | ζ | Stage 5 (PD/TTO) | Predicted | Empirical |
|---|:-:|:-:|:-:|:-:|---:|---:|
| Baseline (PR106x) | ✗ | ✗ | ✗ | ✓ | — | 0.20935 |
| +δ alone | ✓ | ✗ | ✗ | ✓ | TBD | NOT YET DISPATCHED |
| +ε alone | ✗ | ✓ | ✗ | ✓ | TBD | NOT YET DISPATCHED |
| +ζ alone | ✗ | ✗ | ✓ | ✓ | TBD | NOT YET DISPATCHED |
| +δ+ε | ✓ | ✓ | ✗ | ✓ | TBD | NOT YET DISPATCHED |
| +δ+ζ | ✓ | ✗ | ✓ | ✓ | TBD | NOT YET DISPATCHED |
| +ε+ζ | ✗ | ✓ | ✓ | ✓ | TBD | NOT YET DISPATCHED |
| +δ+ε+ζ (full stack) | ✓ | ✓ | ✓ | ✓ | 0.155-0.175 | NOT YET DISPATCHED |

Each row requires a Lightning T4 contest_auth_eval (~$0.30-0.60 per row).
Total ablation cost: ~$2-5 if reused checkpoint cache; ~$5-15 if each requires
its own retrain.

### 4.4 Cross-paradigm composition rows

When XPARADIGM subagent's empirical anchors land at
`feedback_cross_paradigm_hstack_vstack_empirical_anchors_20260508.md`, the
harness adds:

| Row | α (mask) | β (poses) | γ (JCSP) | δ | ε | ζ | Predicted |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---:|
| α-bakeoff winner alone | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | TBD |
| α + γ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | TBD |
| α + β + γ + δ + ε + ζ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 0.140-0.160 [stretch] |

## 5. Code components needed

### Already in place

- `tools/cathedral_autopilot.py` (top-3 recommender; consumed by §3)
- `tools/parallel_dispatch_top_k.py` (concurrent T4 dispatch; consumed by §4.3)
- `tools/harvest_and_reseed.py` (harvest → anchor calibration; consumed in
  feedback loop)
- `submission_archive.py` (archive custody)
- `scripts/remote_archive_only_eval.sh` (canonical T4 contest auth eval)
- `tac.score_geometry` (closed-form contest score analyzer)

### TO BUILD (Phase 4 integration)

- `tools/build_paper_comparison_table.py` — emit the §3 + §4.3 table from
  harvested JSONL + archive manifests. Estimated ~150 LOC.
- `experiments/build_delta_epsilon_zeta_archive.py` (~150 LOC) — Phase 4
  archive assembly. Listed in blueprint §9.
- `scripts/remote_lane_delta_epsilon_zeta.sh` (~100 LOC) — Phase 4 remote
  bootstrap. Listed in blueprint §9.

### Tests required

- `tests/test_paper_harness_table_format.py` — assert output table has all
  required columns + rigor tags
- `tests/test_compliance_check_integration.py` — assert paper rows pass
  `scripts/pre_submission_compliance_check.py --contest-final --strict`

## 6. Disclosure posture (CLAUDE.md "Strategic Secrecy Rule")

Per the Strategic Secrecy Rule:

- The δεζ blueprint and this harness blueprint are PRIVATE until operator
  authorizes disclosure.
- The PR #107 (`apogee`) public disclosure already names the OSS `tac` library
  (`https://github.com/adpena/tac`) per
  `project_pr107_disclosure_tac_oss_public_20260505`.
- comma-lab repo remains PRIVATE; no operational levers (Joint-ADMM allocation
  weights, learned-prior architecture, FiLM-protect patterns) are disclosed
  publicly.
- Cloudflare site URL stays in private docs only per current operator policy.
- Any paper publication must wait for operator approval (Gate 5: "approve
  public submission. IRREVERSIBLE disclosure").

## 7. Outputs (when populated)

The harness will produce:

1. `reports/phase4_paper_comparison_table.json` — machine-readable comparison
2. `docs/paper/04_results.md` updates — replace TBD rows with empirical
3. `reports/phase4_ablation_rows.jsonl` — one JSONL per ablation cell
4. `experiments/results/phase4_delta_epsilon_zeta_<timestamp>/` — per-dispatch
   custody (archive bytes, SHA, contest_auth_eval.json, runtime tree)

## 8. Trigger conditions

This harness becomes ACTIVE when ALL of:

1. Phase 1 audit complete (DONE — see `phase4_optimal_stack_design_20260508_claude.md`)
2. apogee_int6 [contest-CUDA] eval lands (Gate 2 precondition)
3. Operator authorizes Phase 2 GPU spend (Gate 1)
4. arch_shrink_x0.4 Lightning T4 dispatch lands (in flight; ETA 12-18h from
   2026-05-08)

Until then, this blueprint is a structured TODO; no rows are populated and no
score numbers are extracted.

## 9. Cross-references

- `.omx/research/phase4_optimal_stack_design_20260508_claude.md`
- `.omx/research/paradigm_delta_epsilon_zeta_phase1_blueprint_20260507_claude.md`
- `feedback_top3_PRs_are_boltons_on_PR100_substrate_20260507.md`
- `feedback_may_4_hnerv_race_postmortem_20260505.md`
- `feedback_path_b_convergent_findings_summary_20260508.md`
- `project_pr107_disclosure_tac_oss_public_20260505.md`
- `tools/cathedral_autopilot.py`
- `submissions/exact_current/` — current-best submission custody
