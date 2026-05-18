# Codex routing directive: G1 IMMEDIATE-EXECUTION — CPU-axis dominance re-rank via existing dual-eval data
# Date: 2026-05-18
# Authority: PRIMARY rate-attack research memo `.omx/research/rate_attack_43_vectors_meta_paradigm_deep_research_20260518.md` (commit 2cae89a87 estimated)
# Hotz BINDING DIRECTIVE within PRIMARY: "G1 PROCEED IMMEDIATELY ($0 cost, ~50 LOC, can land THIS SESSION via local ranker re-computation on existing PR101+102+103+106+107 dual-eval data)"
# Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" 2026-05-04 update + "Submission auth eval — BOTH CPU AND CUDA" non-negotiable
# G1 is SATURATION-INDEPENDENT (cross-axis gap is source-verified empirical fact independent of A-2 N-7 saturation hypothesis)

## CANONICAL POINTERS

1. `/Users/adpena/Projects/pact/CLAUDE.md` (FULL; especially "SegNet vs PoseNet importance — operating-point dependent" + "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" + Catalog #127 custody validator + Catalog #316 frontier scan)
2. `/Users/adpena/Projects/pact/AGENTS.md`
3. `.omx/research/rate_attack_43_vectors_meta_paradigm_deep_research_20260518.md` (PRIMARY rate-attack master memo; G1 design specifications in its per-TOP-5 section)
4. `.omx/research/rate_attack_vector_g1_*_20260518.md` (per-TOP-5 design memo for G1; produced by PRIMARY subagent a703d2b74784d4f00)
5. `.omx/state/codex_persistent_session_state.jsonl` (Codex's current state)
6. `reports/latest.md` (current frontier; R5-3 reactivation triggered)

## EMPIRICAL FOUNDATION (HARD-EARNED-VERIFIED per Catalog #229)

- PR102: contest-CUDA = 0.22839 vs contest-CPU = 0.19538 → **Δ = +0.033 cross-axis gap**
- Contest LEADERBOARD ranks by contest-CPU
- Other public PR archives (PR101 + PR103 + PR106 + PR107) all have paired CPU+CUDA measurements per `experiments/results/public_pr*_intake_*/`
- Per CLAUDE.md "SegNet vs PoseNet importance" 2026-05-04 update: pose marginal is 2.71× SegNet's at PR106 frontier — but CPU/CUDA divergence is structurally orthogonal to this

## WHAT THIS DIRECTIVE ROUTES

G1 (CPU-axis dominance) is the **simplest highest-EV rate-attack vector** in the PRIMARY's TOP-5. It does NOT require new bytes in the archive — it reranks CHOICES of EXISTING bytes by exploiting the CPU-vs-CUDA score asymmetry on existing public-PR dual-eval data.

### Phase 1: Cross-axis re-ranking helper (~50 LOC)

`tools/probe_g1_cpu_axis_re_rank.py`:

```python
# Read existing dual-eval data for all public PRs:
#   PR101 + PR102 + PR103 + PR106 + PR107
# For each PR, parse the contest-CPU score + contest-CUDA score from
# their respective archive's published eval-comment artifacts (e.g.
# experiments/results/public_pr*_intake_*/archive_eval_*.json or
# the public PR-comment scorecard at tools/public_pr_eval_comment_scorecard.py)

def rank_by_cpu_axis() -> list[dict[str, Any]]:
    """Rank current public-PR archives by contest-CPU score (leaderboard axis).
    Returns sorted list with CPU score + CUDA score + gap.
    """

def identify_cpu_optimal_candidates(*, max_cuda_regression: float = 0.05) -> list[dict[str, Any]]:
    """Identify candidates where contest-CPU is competitive but
    contest-CUDA is significantly worse (suggesting CPU-axis-specific exploit)."""

def find_cpu_axis_re_rank_opportunity() -> dict[str, Any]:
    """Find archives where re-ranking by CPU axis would change the
    "best-archive" verdict vs CUDA-axis ranking.
    Returns: dict with current-CUDA-best vs current-CPU-best + delta."""
```

### Phase 2: Frontier-displacement check via existing artifacts

For each public PR archive, recompute the contest-CPU axis ranking using existing dual-eval data. Verify:
- Does CPU-axis ranking match the leaderboard ranking? (Sanity check)
- Are there archives where CPU-axis score is BETTER than the current local frontier (`0.19205 [contest-CPU]`)?
- What's the predicted CPU-axis score for our PR101_lc_v2 clone (we measured CUDA at `0.20533`; what's the corresponding CPU score?)

### Phase 3: Operator-facing report

`experiments/results/g1_cpu_axis_re_rank_<utc>/report.json`:
```json
{
  "axis_rank_cpu": [...],     // sorted by CPU score ascending
  "axis_rank_cuda": [...],    // sorted by CUDA score ascending
  "axis_gap_per_archive": {...},  // Δ per archive
  "re_rank_opportunities": [...],  // archives where CPU-rank ≠ CUDA-rank
  "predicted_cpu_score_pr101_lc_v2": <float or null if not derivable>,
  "verdict": "FRONTIER_MOVES_VIA_RE_RANK" | "FRONTIER_STABLE_VIA_RE_RANK" | "INSUFFICIENT_DATA",
  "predicted_delta_s_band": [...]
}
```

### Phase 4: Probe outcome registration

Per Catalog #313:
```python
register_probe_outcome(
    probe_id="g1_cpu_axis_re_rank_20260518",
    verdict=<FRONTIER_MOVES_VIA_RE_RANK | FRONTIER_STABLE | INSUFFICIENT_DATA>,
    status="adjudicated",
    rationale=<report_path>,
    expires_at_utc=<+30d>,
    agent="codex",
)
```

## DISCIPLINE

- Catalog #229 premise verification: read actual public-PR eval-comment data before computing
- Catalog #287 evidence tags on every score
- Catalog #206 checkpoint discipline
- Catalog #117/#157/#174 commit serializer with POST-EDIT sha
- Catalog #127 custody validator: scores MUST be tagged with axis + hardware_substrate
- Catalog #313 register probe outcome
- Catalog #245 4-layer pattern N/A (single-shot probe)
- Catalog #314 absorption avoidance: scope is `tools/probe_g1_cpu_axis_re_rank.py` + `experiments/results/g1_cpu_axis_re_rank_*` + `.omx/state/probe_outcomes.jsonl` registration

## EXIT CRITERIA

- [ ] `tools/probe_g1_cpu_axis_re_rank.py` exists; runnable via CLI
- [ ] Report at `experiments/results/g1_cpu_axis_re_rank_<utc>/report.json`
- [ ] Probe outcome appended to `.omx/state/probe_outcomes.jsonl` per Catalog #313
- [ ] codex_persistent_session_state row appended
- [ ] Memory entry `feedback_g1_cpu_axis_re_rank_landed_20260518.md` documenting verdict

## SISTER COORDINATION

In-flight:
- SYNTHESIS-V2 subagent (`a18c228872a761bdb`) reconciling PRIMARY+ADVERSARIAL+supplement; will reference G1 outcome
- A-2 N-7 routing directive (commit `1ac2063de`) — Codex executes for saturation hypothesis test
- Codex session `019de465` continues per /goal LOOP

DISJOINT scope: G1 probe is single-shot read-only analysis on existing eval data. No source-code edits required.

## OPERATOR-FACING NOTE

This routes the **HOTZ BINDING DIRECTIVE** from PRIMARY rate-attack research §G1: "PROCEED IMMEDIATELY". Per Hotz: $0 cost, ~50 LOC, can land THIS SESSION. The verdict will tell the operator whether the current local frontier (`0.19205 [contest-CPU]`) can be displaced via CPU-axis-specific re-ranking of EXISTING public-PR archives (NO new training or dispatch required).

If verdict = FRONTIER_MOVES_VIA_RE_RANK: immediate frontier improvement achievable at $0. Operator can update reports/latest.md (R5-3 closure) + submit improved archive to contest.

If verdict = FRONTIER_STABLE_VIA_RE_RANK: G1's exploit doesn't help the current frontier directly; remains useful for future rate-attack composition.

— Main-Claude 2026-05-18 (PRIMARY+HOTZ-authorized routing per G1 binding directive)
