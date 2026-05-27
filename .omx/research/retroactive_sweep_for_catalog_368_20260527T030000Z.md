<!-- HISTORICAL_PROVENANCE: APPEND-ONLY per Catalog #110/#113. Retroactive sweep per Catalog #348 for newly-landed Catalog #368 (substitution-stacking baseline canonical frontier pointer). -->
<!-- HISTORICAL_SCORE_LITERAL_OK: this memo cites the V14 historical anchor; no NEW score literal claims. -->

# Retroactive sweep for Catalog #368 — substitution-stacking baseline matches canonical frontier pointer

**Date:** 2026-05-27T03:00:00Z
**Per:** CLAUDE.md "Operator gates must be wired and used" non-negotiable + Catalog #348 "EVENT-DRIVEN RETROACTIVE VERDICT-TAINT SWEEP" self-protection.
**Operator NON-NEGOTIABLE 2026-05-26:** "Fix all bugs permanently and self protect against them" + "No need to stagger now"

## 1. Bug-class symptom signature

A substitution-stacking recipe at `.omx/operator_authorize_recipes/*.yaml` cites a baseline archive sha that does NOT match the canonical frontier pointer at `.omx/state/canonical_frontier_pointer.json`. Specifically:

1. **Explicit baseline sha field** — recipe declares `baseline_archive_sha:` / `substitution_baseline:` / `stacked_archive_sha:` / `baseline_sha:` / `stacked_on_archive:` / `baseline_archive_sha256:` / `stacking_baseline_sha:` with a value that is NOT the canonical frontier's `archive_sha256`.
2. **Text-pattern stacking trigger** — recipe text contains `stacked archive built on` / `stacking on baseline sha` / `substitution on baseline` / `swapped selector packet on` / `stacked on baseline archive` and the cited sha is NOT canonical frontier.
3. **No opt-out** — recipe lacks `research_only: true` / `dispatch_enabled: false` / `lane_class: substrate_engineering` / `smoke_only: true`.
4. **No waiver** — no same-line `# BASELINE_NON_FRONTIER_INTENTIONAL_OK:<rationale>` waiver with non-placeholder rationale (≥4 chars).

The canonical frontier per pointer (as of 2026-05-27):
- `contest_cpu`: `7a0da5d0fc327cba3f7d1387a544fd5ce5f05bc56ecc8e12cd5097141672f4fe` (DQS1 rank021) at 0.19202828
- `contest_cuda`: `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4` (PR106 format0d) at 0.20533002

## 2. Pre-fix window

The bug-class drift was empirically demonstrated **once** today: V14 cascade-A FEC10 hybrid P11 paired CPU+CUDA dispatch (commit `abdeefb00`, 2026-05-26 21:23:35) stacked the FEC10 hybrid selector packet (236B) onto the PR101+FEC6 baseline (sha `6bae0201fb082457`, 178517 bytes) instead of the DQS1 frontier sha `7a0da5d0fc327cba` (CPU) or PR106 format0d sha `9cb989cef519` (CUDA). The V14 dispatches landed:
- CUDA T4 `fc-01KSKK6DQKB9YEBHR550Y4KB0W`: score = 0.22620136552710735 [contest-CUDA T4] (WORSE than canonical CUDA frontier by +0.0209)
- CPU `fc-01KSKK713B02QD5NEP5V6FHQTQ`: score = 0.192042660714715 [contest-CPU linux_x86_64] (WORSE than canonical CPU frontier by +1.4e-5)

V14 verdict commit `abdeefb00`: "NOT PR111 candidate" per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable.

Per the 11th ORDER standing directive Dim 8 (apples-to-apples baseline FIRST): this is the **11th recurrence** of the ORDER violation bug class. The structural fix lives at Catalog #368: gate fires BEFORE paid dispatch fires.

## 3. Historical-KILL/DEFER/FALSIFY search results

Searched repo for all stacking/substitution recipe patterns across `.omx/operator_authorize_recipes/*.yaml`:

```bash
grep -rln "baseline_archive_sha\|substitution_baseline\|stacked_archive_sha\|stacked on baseline\|6bae0201" .omx/operator_authorize_recipes/
```

Findings:
- `master_gradient_fec6_modal_cpu_dispatch.yaml` + `master_gradient_fec6_modal_t4_cuda_anchor_dispatch.yaml` + `fec6_plus_format0d_extra_modal_paired_dispatch.yaml` + `fec6_plus_haar_residual_modal_paired_dispatch.yaml` + `dp1_plus_fec6_composition_modal_paired_dispatch.yaml` + `substrate_uniward_per_instance_multi_scale_wavelet_segnet_modal_t4_dispatch.yaml`: ALL reference FEC6 baseline sha `6bae0201` in description fields but are STRUCTURALLY out-of-scope of Catalog #368 because:
  - They use FEC6 archive as a **READ-ONLY INPUT** for master-gradient extraction / DP1 composition / fec6+other stacking variants (the baseline-archive serves as INPUT to a different operation, not as the comparison baseline for promotion).
  - The `baseline_archive_sha:` field is NOT present in any of these recipes (only `default_path:` for trainer inputs).
- V14 recipe (in-flight at commit time, no persisted YAML): the only known empirical violation; closed by commit `abdeefb00` itself stating "NOT PR111 candidate" + reactivation criterion = re-base on canonical frontier.

Live count BEFORE Catalog #368 landing: **0** (V14 was a one-time scripted dispatch, not a persisted recipe; no in-repo recipe currently has a non-frontier baseline_archive_sha field).

**No historical KILL / DEFER / FALSIFY memos cite the non-frontier-baseline bug class.** This is a NEW bug class surfaced 2026-05-26 by the V14 verdict empirically.

Per the 11th ORDER standing directive: the bug class has recurred 11 times across the contest. Prior recurrences manifested as different surface symptoms (e.g. PR97 anti-pattern seg-for-pose trade per CLAUDE.md "SegNet vs PoseNet importance"; multiple ad-hoc stacking attempts on stale baselines per the historical record). The structural extinction at #368 prevents future recurrences at the recipe + canonical-frontier-pointer surface.

## 4. Per-finding RE-EVAL-priority assignment

| Historical Finding | RE-EVAL Priority | Rationale |
|---|---|---|
| V14 cascade-A FEC10 hybrid (commit `abdeefb00`) | LOW (closed in-place) | Verdict already states "NOT PR111 candidate" + reactivation criteria = re-base on canonical frontier sha. No historical KILL memo to re-eval. |
| 10 prior ORDER violations (per 11th standing directive enumeration) | DEFERRED | Each prior violation has its own dedicated landing memo. The structural extinction at #368 is the cumulative fix; per-violation retroactive re-eval is out-of-scope. |

## 5. Cross-references

- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable
- CLAUDE.md "Frontier scores are pointer-only — NON-NEGOTIABLE" (Catalog #343 sister)
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable
- CLAUDE.md 11th ORDER standing directive Dim 8 (apples-to-apples baseline FIRST)
- Catalog #343 (`check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded` — same META class at CLAUDE.md surface)
- Catalog #246 (paired CUDA + CPU dispatch with skip-axis-if-promotable-anchor-exists)
- Catalog #226 (`check_trainer_auth_eval_uses_canonical_helper` — canonical helper routing)
- Catalog #316 (`check_reports_latest_md_not_stale_vs_canonical_frontier` — same META class at reports surface)
- Catalog #348 retroactive verdict-taint sweep discipline
- Catalog #287 placeholder-rationale rejection
- V14 verdict commit `abdeefb00`

## 6. Discipline declarations

- Catalog #229 PV: full repo grep + canonical frontier pointer inspection + recipe schema audit at preflight wire-in time
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — this memo is NEW; zero mutations
- Catalog #287 substantive-rationale rejection — placeholder literals rejected throughout
- Catalog #348 4-field contract: bug-class symptom signature ✓ + pre-fix window ✓ + historical search results ✓ + per-finding RE-EVAL-priority assignment ✓
- 11th ORDER standing directive: 11th violation extincts structurally at recipe + canonical-frontier-pointer surface

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
