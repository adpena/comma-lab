---
schema: codex_routing_directive_v1
directive_id: codex_routing_directive_rate_attack_vector_2_g1_cpu_axis_optimization_20260518
target_subagent: codex_019de465
routing_date: "2026-05-18"
parent_design_memo: rate_attack_vector_2_g1_cpu_axis_optimization_design_memo_20260518
parent_master_memo: rate_attack_43_vectors_meta_paradigm_deep_research_20260518
meta_paradigm_anchor: structural_information_not_shipped_meta_paradigm_unification_20260518
vector_id: G1
priority: TOP-2_IMMEDIATE
council_verdict: PROCEED_IMMEDIATELY_HOTZ_BINDING
binding_revisions:
  - "Hotz: PROCEED IMMEDIATELY (zero GPU cost; pure re-ranking on existing dual-eval data)"
operator_approved_gpu_budget_usd: 0.00
write_scope_for_codex: |
  tools/cpu_axis_optimal_archive_selector.py  (NEW canonical helper)
  tools/scan_best_anchor_per_axis.py  (EXTENSION; existing per Catalog #316)
  reports/latest.md  (UPDATE FRONTIER section with G1 verdict)
  src/tac/tests/test_cpu_axis_optimal_archive_selector.py  (NEW tests)
write_scope_excludes:
  - "Anything in PRIMARY research subagent scope"
  - "Anything in ADVERSARIAL sister subagent scope"
---

# Codex Routing Directive — Rate-Attack Vector 2: G1 CPU-Axis-Specific Optimization

**Target Codex subagent**: `019de465`
**Priority**: TOP-2 IMMEDIATE per Hotz binding directive (zero GPU cost; can land THIS SESSION)
**META-paradigm**: SINS — exploits the empirical PR102 +0.033 CPU-CUDA gap. Leaderboard ranks by CPU.

## 0. PRE-FLIGHT

1. Read CLAUDE.md (FULL); honor "Submission auth eval — BOTH CPU AND CUDA" non-negotiable + "SegNet vs PoseNet importance — operating-point dependent" + Catalog #316
2. Read AGENTS.md + MEMORY.md top-50
3. Read parent design memo + parent master memo + META-paradigm
4. Read existing canonical: `tools/scan_best_anchor_per_axis.py` + `src/tac/frontier_scan.py`
5. Read existing dual-eval data sources:
   - `.omx/state/continual_learning_posterior.jsonl`
   - `.omx/state/modal_call_id_ledger.jsonl`
   - `.omx/state/cost_band_posterior.jsonl`
   - `reports/latest.md` Catalog #316 FRONTIER section
6. Lane registry check: `lane_rate_attack_g1_cpu_axis_specific_20260518` already pre-registered at L0

## 1. Phase 1 — LOCAL RE-RANKING ($0; CAN LAND THIS SESSION)

### 1.1 Land `tools/cpu_axis_optimal_archive_selector.py` (~80 LOC)

**Goal**: Scan all dual-eval data; for each archive family (pr101_*, pr102_*, pr103_*, pr106_*, pr107_*), compute CPU-axis-min archive; compare to current frontier.

**Pseudocode**:
```python
#!/usr/bin/env python3
"""
G1 canonical helper per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" + Catalog #316.

Per Hotz binding: PROCEED IMMEDIATELY. Leaderboard ranks by CPU. Re-rank existing
archives by CPU axis ONLY; emit operator-facing verdict.
"""
import argparse, json, re, sys
from pathlib import Path
from collections import defaultdict
from tac.frontier_scan import Anchor, collect_all_anchors, best_per_axis

REPO_ROOT = Path(__file__).resolve().parents[1]

# Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" QUALIFYING_HARDWARE
QUALIFYING_HARDWARE_CPU = ("linux_x86_64_cpu",)

def family_of(archive_id):
    # Group archives by family prefix
    for prefix in ("pr101_", "pr102_", "pr103_", "pr106_", "pr107_"):
        if prefix in archive_id:
            return prefix.rstrip("_")
    return "other"

def select_cpu_optimal_per_family(anchors):
    """For each family, select the anchor with the MIN CPU score (since lower=better)."""
    cpu_anchors = [a for a in anchors if a.hardware_substrate in QUALIFYING_HARDWARE_CPU]
    by_family = defaultdict(list)
    for a in cpu_anchors:
        by_family[family_of(a.archive_id)].append(a)
    per_family_optimal = {
        fam: min(a_list, key=lambda a: a.score)
        for fam, a_list in by_family.items()
    }
    return per_family_optimal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    ap.add_argument("--current-frontier-cpu", type=float, default=0.19205)
    ap.add_argument("--json", action="store_true", help="emit JSON output")
    args = ap.parse_args()

    payload = build_frontier_scan_payload(args.repo_root)  # existing canonical helper
    all_anchors = payload.anchors

    per_family = select_cpu_optimal_per_family(all_anchors)

    # Find the overall CPU-axis-min
    overall_cpu_min = min(per_family.values(), key=lambda a: a.score)

    delta_vs_current = overall_cpu_min.score - args.current_frontier_cpu

    result = {
        "current_frontier_cpu": args.current_frontier_cpu,
        "g1_cpu_optimal_archive_id": overall_cpu_min.archive_id,
        "g1_cpu_optimal_score": overall_cpu_min.score,
        "g1_predicted_delta": delta_vs_current,
        "per_family_optimal": {fam: {"archive_id": a.archive_id, "score": a.score} for fam, a in per_family.items()},
        "evidence_grade": "[contest-CPU]",
        "score_claim_valid": True,
        "axis_score": "contest_cpu",
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"G1 CPU-axis optimal archive: {overall_cpu_min.archive_id}")
        print(f"  Score: {overall_cpu_min.score} [contest-CPU]")
        print(f"  Delta vs current frontier ({args.current_frontier_cpu}): {delta_vs_current:+.5f}")
        print(f"  Per-family: {per_family}")

    return 0 if delta_vs_current < -0.003 else 1  # rc=0 if within predicted band

if __name__ == "__main__":
    sys.exit(main())
```

### 1.2 Extend `tools/scan_best_anchor_per_axis.py` with G1 hook

Existing helper handles overall best-per-axis. Extend to emit per-family CPU-axis breakdown.

### 1.3 Update `reports/latest.md` FRONTIER section with G1 verdict

Per Catalog #316: the frontier section MUST cite current CPU/CUDA best AND any G1-discovered re-ranking opportunities.

Append after existing FRONTIER section:

```markdown
## G1 CPU-Axis Optimization (per master memo TOP-2)

Per Hotz binding directive 2026-05-18: leaderboard ranks by CPU.

Per-family CPU-axis-optimal archives:

| Family | Archive | CPU score |
|---|---|---|
| pr101 | pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean | 0.19205 |
| pr102 | (pending re-eval) | (pending) |
| pr103 | (pending re-eval) | (pending) |
| pr106 | (pending re-eval) | (pending) |
| pr107 | pr107_apogee (M5 Max advisory match GHA Linux x86_64) | 0.19664 |

G1-discovered CPU-optimal: pending Phase 1 run.

If G1-optimal differs from current 0.19205 by > -0.003 → submit per CLAUDE.md "Submission PR gate" non-negotiable.
```

## 2. Phase 2 — PAIRED Linux x86_64 [contest-CPU] RE-EVAL ($0; ONLY IF Phase 1 finds improvement)

If Phase 1 identifies a CPU-optimal archive different from current 0.19205:

1. Verify the candidate archive has a CURRENT paired Linux x86_64 [contest-CPU] anchor (NOT macOS-CPU advisory per Catalog #192)
2. If only macOS-CPU advisory exists → run paired Linux x86_64 re-eval via Vast.ai CPU / Modal CPU / GHA CI
3. Confirm score within predicted band

## 3. Phase 3 — SUBMIT PR (if validated)

Per CLAUDE.md "Submission PR gate" non-negotiable: 5-turn consecutive clean-pass adversarial skunkworks council review BEFORE PR.

## 4. Discipline (NON-NEGOTIABLE)

- Catalog #229 premise verification (the PR102 +0.033 anchor is HARD-EARNED-VERIFIED per CLAUDE.md)
- Catalog #287 evidence tags (use `[contest-CPU]` for Linux x86_64 + `[macOS-CPU advisory]` for M5 Max — NEVER conflate)
- Catalog #117/#157/#174 commit serializer
- Catalog #126 lane already pre-registered
- Catalog #316 frontier ledger update + Catalog #245 Modal call_id ledger if any re-eval landed

## 5. Cross-References

- Parent design memo + parent master memo + META-paradigm
- Catalog #316 canonical: `tools/scan_best_anchor_per_axis.py` + `src/tac/frontier_scan.py`
- CLAUDE.md non-negotiables: "Submission auth eval — BOTH CPU AND CUDA" / "Forbidden score claims" / "MPS auth eval is NOISE" / "SegNet vs PoseNet importance — operating-point dependent"
- Catalog gates: #127 / #192 / #205 / #316 / #324 / #287 / #245

## 6. Closeout

G1 is FREE leaderboard-ranking improvement. Can land THIS SESSION via local re-ranking. Hotz binding directive: PROCEED IMMEDIATELY.

**Expected return**: -0.003 to -0.010 ΔS on CPU axis. **Even the lower bound at $0 cost is highest-EV per-dollar of the entire TOP-5.**
