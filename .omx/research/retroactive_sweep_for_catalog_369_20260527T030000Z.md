<!-- HISTORICAL_PROVENANCE: APPEND-ONLY per Catalog #110/#113. Retroactive sweep per Catalog #348 for newly-landed Catalog #369 (inflate consumes real trained weights not synthetic frame base). -->
<!-- HISTORICAL_SCORE_LITERAL_OK: this memo cites the Cascade C' WAVE-6 historical anchor; no NEW score literal claims. -->

# Retroactive sweep for Catalog #369 — inflate consumes real trained weights not synthetic frame base

**Date:** 2026-05-27T03:00:00Z
**Per:** CLAUDE.md "Operator gates must be wired and used" non-negotiable + Catalog #348 "EVENT-DRIVEN RETROACTIVE VERDICT-TAINT SWEEP" self-protection.
**Operator NON-NEGOTIABLE 2026-05-26:** "Fix all bugs permanently and self protect against them" + 11th ORDER standing directive Dim 2 (operation ordering within pipeline)

## 1. Bug-class symptom signature

A substrate inflate.py file at `submissions/*/inflate.py` (excluding `submissions/exact_current/` per CLAUDE.md mutation frontier) or `src/tac/substrates/*/inflate.py` generates synthetic frame bases without consuming real trained weights derived from real-video input. Specifically:

1. **Synthetic frame function pattern** — file contains one of: `_render_frame_0_base`, `_synthesize_frame_base`, `_render_frame_placeholder`, `_render_frame_synthetic`, `synthetic_frame_for_per_pair_warp`, `_render_synthetic_rgb`, `_synthetic_base_frame`, `render_deterministic_textured_RGB`, `deterministic_textured_rgb`, `synthesize_textured_frame_base`.
2. **Synthetic operation indicators** — file contains operational patterns like sinusoidal grid (`np.sin(xs * (4.0 * np.pi / width))`), radial gradient (`radial = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)`), or hardcoded mean+amplitude RGB channels (e.g. `(96 + 96 * gx).astype(np.uint8)`).
3. **No real-frame-vendor token** — file does NOT import / use one of: `Comma2k19LocalCache.fetch_chunk` (per Catalog #213 canonical), `frame_0_from_archive_bytes` sister pattern, `real_frame_from_pyav`, `av.open` (pyav for real-video decode), `decode_real_frame_from`, `extract_real_frame_0`, `real_dashcam_frame`, `canonical_real_frame_loader`.
4. **No waiver** — no same-line `# SYNTHETIC_FRAME_BASE_INTENTIONAL_OK:<rationale>` waiver with non-placeholder rationale (≥4 chars).
5. **No recipe opt-out** — adjacent substrate recipe at `.omx/operator_authorize_recipes/substrate_<substrate_id>_*.yaml` does NOT declare `research_only: true` / `dispatch_enabled: false` / `smoke_only: true` / `lane_class: substrate_engineering`.

## 2. Pre-fix window

The bug-class was empirically demonstrated **once today** by Cascade C' WAVE-6 (commit `e215ee555`, 2026-05-26 21:32:56): Modal T4 dispatch `fc-01KSKK64ERJW0KSFB5T905RBJC` landed `score=85.43 [diagnostic_cpu]` vs canonical frontier 0.192028 = **445× WORSE**. The inflate.py at `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/inflate.py:183-223` uses synthetic `_render_frame_0_base()` (sinusoidal R/G + radial B; mean=142.25 std=30.02) instead of real frame_0 from archive bytes.

The Cascade C' WAVE-6 verdict commit explicitly states paradigm INTACT per Catalog #307 (IMPLEMENTATION-LEVEL not PARADIGM-LEVEL falsification) and queues WAVE-7 multi-axis debug reactivation path #1: "vendor real frame_0 reference (resolves Catalog #220 SCAFFOLD_DEFERRED_INTEGRATION)". This Catalog #369 gate makes that operator-routable path STRUCTURAL — future inflate.py files cannot silently regress to synthetic frame base patterns.

Per the 11th ORDER standing directive Dim 2 (operation ordering within pipeline): the canonical operational order is **trainer-FIRST consumes real Comma2k19 dashcam frames → trainer emits real learned weights → inflate-SECOND consumes those weights to reconstruct real frames**. Inflate-side synthetic frame generation breaks this order because inflate is operating WITHOUT trainer-emitted weights as input.

## 3. Historical-KILL/DEFER/FALSIFY search results

Searched repo for all inflate.py files with synthetic frame patterns:

```bash
grep -l "synthetic\|sinusoidal_R_G\|_render_frame_0_base\|deterministic_textured_RGB\|radial_B" \
    submissions/*/inflate.py src/tac/substrates/*/inflate.py
```

Findings (1 file):
- `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/inflate.py` — the canonical bug-class anchor (4 synthetic-pattern matches). Recipe currently has `dispatch_enabled: true` + `research_only: false` (per commit `aaf0b1eb6` Phase 2 council symposium). Gate flags it as the expected baseline; operator-routable: either (a) vendor real frames per Catalog #213, (b) add `# SYNTHETIC_FRAME_BASE_INTENTIONAL_OK:<rationale>` waiver, OR (c) flip recipe to `research_only: true` per Catalog #240.

Live count BEFORE Catalog #369 landing: **1** (Cascade C' is the canonical bug-class anchor; expected per CLAUDE.md "Strict-flip atomicity rule"). The gate is wired WARN-ONLY initially per the rule; strict-flip planned when Cascade C' either lands the real-frame vendor OR opts out via waiver/recipe.

**No historical KILL / DEFER / FALSIFY memos cite synthetic-frame-base inflate bug class.** This is a NEW bug class surfaced 2026-05-26 by Cascade C' WAVE-6 verdict empirically.

Sister anchor: NSCS06 v8 chroma_lut pattern (per Cascade C' WAVE-5 inflate.py docstring) IS the canonical real-frame-vendor pattern — derives frame_0 from a chroma_lut lookup over real grayscale+cls inputs shipped in the archive. Catalog #369's acceptance cascade (a) recognizes this pattern via `frame_0_from_archive_bytes` sister token.

## 4. Per-finding RE-EVAL-priority assignment

| Historical Finding | RE-EVAL Priority | Rationale |
|---|---|---|
| Cascade C' WAVE-6 verdict (commit `e215ee555`) | LOW (live anchor; gate baseline) | Verdict already classifies IMPLEMENTATION-LEVEL falsification per Catalog #307. Operator-routable reactivation path = vendor real frames per Catalog #213 (gate accepts immediately on first vendor-token import). Gate is WARN-ONLY pending the WAVE-7 fix. |
| WAVE-4 all-zero placeholder (superseded by WAVE-5 `cfed4dc10`) | N/A (already superseded) | The all-zero placeholder pattern was already extincted by WAVE-5; Catalog #369 catches the WAVE-5 synthetic-textured replacement which is a DIFFERENT bug class within the same META-class (inflate-side frame generation without trainer-emitted weights). |
| Sister NSCS06 v8 chroma_lut pattern | N/A (canonical reference) | NSCS06 v8 is the canonical real-frame-from-archive pattern that Cascade C' WAVE-7 should adopt. No KILL/DEFER memo applies. |

## 5. Cross-references

- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable
- CLAUDE.md "Forbidden premature KILL without research exhaustion" (Cascade C' paradigm INTACT)
- CLAUDE.md 11th ORDER standing directive Dim 2 (operation ordering within pipeline)
- Catalog #213 (`check_comma2k19_downloads_route_through_canonical_cache` — canonical real-frame vendor)
- Catalog #146 (`check_phase1_trainer_runtime_emits_contest_compliant_inflate` — contest-compliant inflate runtime)
- Catalog #205 (`check_inflate_py_uses_canonical_select_inflate_device` — canonical inflate device-fork)
- Catalog #220 (`check_substrate_l1_scaffold_no_byte_addition_without_operational_score_improvement_mechanism` — operational mechanism)
- Catalog #272 (`check_substrate_distinguishing_feature_integration_contract` — distinguishing-feature integration)
- Catalog #307 (`check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification` — Cascade C' WAVE-6 explicitly cites this)
- Catalog #240 (`check_substrate_contest_cuda_chain_complete_or_research_only_tagged` — recipe-vs-trainer chain consistency)
- Catalog #348 retroactive verdict-taint sweep discipline
- Catalog #287 placeholder-rationale rejection
- Cascade C' WAVE-6 verdict commit `e215ee555` + WAVE-5 inflate fix `cfed4dc10`

## 6. Discipline declarations

- Catalog #229 PV: full repo grep on `submissions/*/inflate.py` + `src/tac/substrates/*/inflate.py` + recipe schema audit at preflight wire-in time
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — this memo is NEW; zero mutations
- Catalog #287 substantive-rationale rejection — placeholder literals rejected throughout
- Catalog #348 4-field contract: bug-class symptom signature ✓ + pre-fix window ✓ + historical search results ✓ + per-finding RE-EVAL-priority assignment ✓
- 11th ORDER standing directive Dim 2: structural extinction at inflate.py source-text surface

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
