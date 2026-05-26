<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE — this memo is the canonical R1 AGGREGATE review record for Path 3 candidates A + D + E (DreamerV3 RSSM + Z6 predictive-coding + BoostNeRV-against-PR110 MLX-local L0 SCAFFOLDS landed 2026-05-26). DO NOT mutate after landing. -->
<!-- Catalog #344 canonical equation cross-ref: cross-substrate META review across 3 sister R1 memos; FIX-WAVE-R1 op-routable queue priority-ranked. FORMALIZATION_PENDING:r1_aggregate_methodology_pending_council_revisions_per_recursive_adversarial_review_protocol_close_paths -->
---
council_tier: T3
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Tao
  - Carmack
  - Hotz
  - Quantizr
  - MacKay
  - Selfcomp
  - Ballé
  - Hassabis
  - PR95Author
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "T3 tier is appropriate for an AGGREGATE review spanning 3 sister landings"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'Council hierarchy: 4-tier protocol' Tier elevation triggers T2→T3 trigger (a): the aggregate finding touches a CLAUDE.md non-negotiable (Recursive adversarial review protocol — close paths — items 1-8 + new item 8 assumption-challenge axis). T3 cadence budget ≤3/week per CLAUDE.md is within bounds; this is the only T3 review this week."
  - assumption: "Cross-substrate META findings warrant a CONSOLIDATE-OP queued for L1+ rather than blocking R2"
    classification: HARD-EARNED
    rationale: "The META class of 'locally invented MLX primitive diverges from sister-canonical' (A=DreamerV3 PixelShuffle bug) is operator-routable to a CONSOLIDATE-OP that extracts BOTH `_pixel_shuffle_2x_nhwc` AND general `_bilinear_resize_nhwc` to canonical `tac.local_acceleration.pr95_hnerv_mlx`. The CONSOLIDATE-OP is a follow-on PER CLAUDE.md 'Subagent coherence-by-default' standing directive 'consolidate everything into META layer'; it does NOT block FIX-WAVE-R1 (which patches A=DreamerV3 in-place by copying D=Z6's correct convention)."
  - assumption: "FIX-WAVE-R1 successor subagent can land all 3 substrates' fixes in a single commit batch"
    classification: HARD-EARNED
    rationale: "All R1 findings are TIGHTLY SCOPED: A=DreamerV3 has 2 code-fix op-routables (module.py edits only) + 1 test threshold update; E=BoostNeRV has 3 documentation-fix op-routables (memo + 2 docstrings); D=Z6 has 0 findings. Total touch surface: ≤6 files; ≤200 LOC of edits. Sister-coherence per Catalog #230 ownership map: zero overlap with the 5 in-flight Path 3 candidates B'/C'/F/G/H."
council_decisions_recorded:
  - "Aggregate R1 verdict: NOT CLEAN (2 of 3 substrates require FIX-WAVE-R1; the third (D=Z6) passes cleanly)"
  - "Counter resets to 0 per CLAUDE.md 'Recursive adversarial review protocol — close paths' item 3"
  - "FIX-WAVE-R1 successor subagent required BEFORE R2 fires per protocol item 4"
  - "Priority-ranked FIX-WAVE-R1 op-routable queue: see §FIX-WAVE-R1 op-routable queue below"
  - "R2 readiness verdict: BLOCKED until FIX-WAVE-R1 lands and re-runs (a) the 11 A=DreamerV3 tests post-fix + (b) the 25 E=BoostNeRV tests post-doc-fix + (c) re-measures A=DreamerV3 decoder parity max_abs"
  - "CONSOLIDATE-OP for L1+ (advisory): extract `_pixel_shuffle_2x_nhwc` + general `_bilinear_resize_nhwc` to canonical `tac.local_acceleration.pr95_hnerv_mlx` per CLAUDE.md 'consolidate into META layer' standing directive"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
canonical_equation_refs:
  - categorical_posterior_capacity_vs_continuous_gaussian_v1
  - categorical_blahut_arimoto_rate_distortion_v1
  - procedural_predictor_plus_residual_correction_savings_v1
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - path_3_a_recursive_adversarial_review_r1_3_axis_20260526
  - path_3_d_recursive_adversarial_review_r1_3_axis_20260526
  - path_3_e_recursive_adversarial_review_r1_3_axis_20260526
  - dreamer_v3_rssm_mlx_scaffold_landed_20260526
  - z6_predictive_coding_mlx_scaffold_landed_20260526
  - path_3_e_boost_nerv_against_pr110_substrate_design_20260526
---

# R1 Recursive Adversarial Review — AGGREGATE across Path 3 candidates A + D + E

**Per binding operator directive 2026-05-26**: *"we also need adversarial review against all landing recursive for math and scientific and engineering rigor and for MLX drift minimization and portability via numpy"*

**Per CLAUDE.md "Recursive adversarial review protocol — close paths"**: Round 1 of 3 consecutive clean-pass cycles required before code is cleared for deployment.

**Aggregate R1 verdict**: **NOT CLEAN — counter resets to 0 per protocol item 3**

**Cost**: $0 GPU; ~3-4h wall-clock (per-substrate review 1-1.5h each + aggregate synthesis ~45min)

---

## Per-landing R1 verdict summary

| Landing | Commit | Substrate path | R1 Verdict | Counter advance? | Findings count | FIX-WAVE-R1 required? |
|---|---|---|---|---|---|---|
| **A=DreamerV3 RSSM** | `69253a1cc` | `src/tac/substrates/dreamer_v3_rssm/` | **PROCEED_WITH_REVISIONS** | NO (counter resets) | 2 CRITICAL Axis 2 + 1 P2 Axis 1 documentation | YES — 4 op-routables (2 code-fix + 1 test + 1 doc) |
| **D=Z6 predictive-coding** | `83b9ee3e2` | `src/tac/substrates/time_traveler_l5_z6/` (MLX additions) | **PROCEED — R1 CLEAN PASS** | YES (1/3 advance for this landing) | 0 R1 findings; 3 advisories for L1+ | NO |
| **E=BoostNeRV against PR110** | `83910e54e` | `src/tac/substrates/boost_nerv_pr110_residual/` | **PROCEED_WITH_REVISIONS** | NO (counter resets) | 1 documentation-only finding (BPR1 header size inconsistency); 2 advisory | YES — 5 op-routables (all documentation fixes) |

**Aggregate**: 2/3 substrates require FIX-WAVE-R1 successor. Per CLAUDE.md "Recursive adversarial review protocol — close paths" item 3: a round with ANY issue resets the counter to 0. Per item 4: FIX-WAVE-R1 successor subagent required BEFORE R2 fires.

---

## Cross-substrate META findings

### META finding #1: locally-invented MLX primitives diverge from sister-canonical (A=DreamerV3 PixelShuffle bug class)

**Class**: When two sister substrates BOTH implement a "canonical" primitive locally instead of routing through a SHARED canonical helper, divergence is inevitable.

**Empirical anchor**:
- A=DreamerV3 `_pixel_shuffle_2x_nhwc` (module.py:184-197): uses `transpose (0, 1, 3, 2, 4, 5)` channel-LAST convention → **2.4 absolute drift vs PyTorch**
- D=Z6 `_pixel_shuffle_2x_nhwc` (mlx_renderer.py:361-372): uses `transpose (0, 1, 4, 2, 5, 3)` channel-FIRST convention → **0.0 drift vs PyTorch**

Both substrates have the same intent (mirror `nn.PixelShuffle(2)` on NHWC tensors); the implementations diverge because no shared canonical helper exists yet. The CONSOLIDATE-OP for L1+ is to extract D=Z6's correct impl to `tac.local_acceleration.pr95_hnerv_mlx` (or sister `tac.local_acceleration.mlx_pixel_shuffle`) so future MLX scaffolds inherit ONE source of truth.

**Sister CLAUDE.md anchor**: "Subagent coherence-by-default" standing directive 2026-05-15 verbatim *"consolidate everything into META layer or canonical helpers"*; Catalog #299 quota brake "stop and consolidate" pause discipline; Catalog #335 cathedral consumer canonical contract auto-discovery (same META pattern at the cathedral-consumer surface).

**Op-routable** (queued for L1+; NOT blocking FIX-WAVE-R1): extract `_pixel_shuffle_2x_nhwc` + general `_bilinear_resize_nhwc` to canonical `tac.local_acceleration.pr95_hnerv_mlx`; refactor A=DreamerV3 + D=Z6 + future Path 3 candidates to import from canonical.

### META finding #2: documentation-source-of-truth drift (E=BoostNeRV BPR1 header size inconsistency)

**Class**: When a constant is declared in CODE (single source of truth) but documented in MULTIPLE PLACES (design memo + module docstring + comment), the docs drift from the code over time.

**Empirical anchor** (E=BoostNeRV):
- Source of truth: `__init__.py` `BPR1_HEADER_LEN = 29` matches `struct.calcsize('<5sBBB16sIB') = 29` ✓
- Design memo claims: "24-byte header" (WRONG by 5 bytes)
- archive.py module docstring claims: "BPR1 header 28 bytes" (WRONG by 1 byte)
- __init__.py comment claims: "24-byte header" (WRONG by 5 bytes; SAME FILE as the correct constant)

Sister CLAUDE.md anchor: "Comment-only contracts are FORBIDDEN" non-negotiable. The forensic-debug consequence is the operator-readable struct format `<5sBBB16sIB` documented inconsistently across the 3 surfaces while the actual canonical `BPR1_HEADER_LEN` is correct.

**Op-routable** (P0 FIX-WAVE-R1 — documentation only): 3 string edits across 3 files (memo + docstring + comment) to declare 29 bytes consistently.

### META finding #3: shared MLX-primitive correctness invariant — D=Z6 is the SISTER-CANONICAL reference

**Class**: Across the 3 reviewed substrates, D=Z6's MLX primitives are the only ones empirically PyTorch-byte-stable across the full primitive set (PixelShuffle + bilinear + Linear + Conv2d + activations + stack/reshape/transpose).

**Empirical anchor**: D=Z6 `test_b03` + `test_f03` jointly prove (a) state_dict round-trip + (b) MLX-built archive inflates byte-stably to 24,416,064 bytes via canonical PyTorch inflate runtime. This makes D=Z6 the de-facto canonical reference for the FIX-WAVE-R1 patch to A=DreamerV3 (copy D=Z6's `_pixel_shuffle_2x_nhwc` verbatim).

**No op-routable**: this is a positive META finding (sister-canonical reference); op-routables sit on A=DreamerV3 to align with D=Z6's correct convention.

### META finding #4: canonical equation registry empirically queried — all 3 cited equations REGISTERED

**Class**: Canonical equation citations in landing memos can be EMPIRICALLY VERIFIED via `tac.canonical_equations.query_equations()`. This review queried all 3 substrates' cited equations:

| Substrate | Cited canonical equation | REGISTERED status |
|---|---|---|
| A=DreamerV3 | `categorical_posterior_capacity_vs_continuous_gaussian_v1` | ✓ REGISTERED |
| A=DreamerV3 | `categorical_blahut_arimoto_rate_distortion_v1` | ✓ REGISTERED |
| E=BoostNeRV | `procedural_predictor_plus_residual_correction_savings_v1` | ✓ REGISTERED |
| E=BoostNeRV (memo cites) | `residual_hybrid_boosting_savings_v1` (FORMALIZATION_PENDING placeholder) | NOT REGISTERED (placeholder name; same conceptual entity as the REGISTERED `procedural_predictor_plus_residual_correction_savings_v1`) |
| D=Z6 | (none declared in frontmatter — sister-substrate-wide gap; not this scaffold's responsibility) | — |

**Op-routable** (P2 advisory, NOT blocking FIX-WAVE-R1): E=BoostNeRV memo cleanup to cite the registered equation name `procedural_predictor_plus_residual_correction_savings_v1` instead of the placeholder `residual_hybrid_boosting_savings_v1`.

### META finding #5: training-invalidating MLX bugs surface ONLY at the PyTorch-export boundary

**Class**: A=DreamerV3's MLX bugs (broken PixelShuffle + broken bilinear) are SILENT at MLX-trainer-only smoke time but TRAINING-INVALIDATING at PyTorch-inflate time.

**Mechanism**:
- MLX trainer optimizes against `MLX_buggy_decoder(weights) → MLX_frames`
- PyTorch inflate uses CORRECT canonical primitives: `nn.PixelShuffle(2)` + `F.interpolate(mode='bilinear', align_corners=False)`
- After MLX training converges + state_dict exports via the `_torch_conv_to_mlx`-inverse transpose, the rendered frames at PyTorch-inflate time DO NOT MATCH the frames the MLX trainer observed at convergence
- The L0 smoke trainer "loss decreased 0.13%" on synthetic random targets does NOT reveal this because the targets are noise; the bug surfaces structurally at L1+ score-aware-loss training where the trainer optimizes against scorer feedback dependent on canonical decoder forward semantics

**Sister Catalog anchor**: CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L9 (Runtime closure: same-runtime source replay required); the MLX-trained-PyTorch-inflated model is NOT the same runtime as the MLX-trained-MLX-inflated model.

**Sister #1265 gate implication**: the canonical Catalog #1265 contest-equivalence gate's threshold (`|S_MLX − S_PyTorch| ≤ 0.001 contest-units`) will FAIL on A=DreamerV3 archives until FIX-WAVE-R1 lands. The L0 sister gate built per A=DreamerV3 op-routable #2 (RSSMC1 grammar gate) cannot PASS until the underlying decoder forward semantics match.

**Op-routable**: covered by A=DreamerV3 FIX-WAVE-R1 P0 op-routables (already enumerated above).

---

## FIX-WAVE-R1 op-routable queue (priority-ranked)

Per CLAUDE.md "Recursive adversarial review protocol — close paths" item 2: all FIX-WAVE-R1 issues land in a successor subagent + are committed BEFORE R2 begins.

### P0 / CRITICAL / TRAINING-INVALIDATING (A=DreamerV3 only)

1. **A-OP1**: rewrite `src/tac/substrates/dreamer_v3_rssm/module.py::_pixel_shuffle_2x_nhwc` from channel-LAST convention `(B, H, W, 2, 2, out_C)` + transpose `(0, 1, 3, 2, 4, 5)` to channel-FIRST convention `(B, H, W, out_C, 2, 2)` + transpose `(0, 1, 4, 2, 5, 3)` matching the SISTER-CANONICAL impl in D=Z6 (`src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py::_pixel_shuffle_2x_nhwc`). Empirically verified: D=Z6's convention is byte-stable vs PyTorch reference.

2. **A-OP2**: replace `src/tac/substrates/dreamer_v3_rssm/module.py::_bilinear_resize_2x_nhwc` (mx.repeat 2x) with import + usage of canonical `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc`. Empirically verified: canonical helper is byte-stable vs PyTorch reference.

### P0 / DOCUMENTATION FIX (E=BoostNeRV only)

3. **E-OP1**: update design memo `.omx/research/path_3_e_boost_nerv_against_pr110_substrate_design_20260526.md` §"Archive grammar" + §"BPR1 sidecar magic, prepended to PR110 archive bytes" rows to declare 29-byte header (currently "24-byte header")

4. **E-OP2**: update `src/tac/substrates/boost_nerv_pr110_residual/archive.py` module docstring line 8 from "BPR1 header 28 bytes" to "BPR1 header 29 bytes"

5. **E-OP3**: update `src/tac/substrates/boost_nerv_pr110_residual/__init__.py` line 41 archive grammar comment from "24-byte header" to "29-byte header"

### P1 / VERIFICATION (A=DreamerV3 post-fix)

6. **A-OP3**: after A-OP1 + A-OP2 land, re-measure `test_mlx_pytorch_decoder_parity_at_archive_boundary` and tighten threshold from `< 50.0` to `< 5.0` (or whatever the post-fix empirical anchor reveals — likely `< 1.0` per empirical individual-primitive drift measurements)

### P2 / DOCUMENTATION (A=DreamerV3)

7. **A-OP4**: amend `.omx/research/dreamer_v3_rssm_mlx_scaffold_landed_20260526.md` cargo-cult audit row #5 (NHWC↔NCHW conv layout transpose correctness-preserving) — the empirical verification cited holds for SHAPE correctness only; FULL decoder forward equivalence requires the A-OP1 + A-OP2 fixes first. Add APPEND-ONLY footer with FIX-WAVE-R1 verdict (or per Catalog #110 APPEND-ONLY: add a sister "POST-R1 amendment" memo at `.omx/research/dreamer_v3_rssm_mlx_scaffold_post_r1_amendment_20260526.md`).

### P2 / ADVISORY (E=BoostNeRV)

8. **E-OP4**: update design memo §"Cross-references" canonical-equation-registry line to cite the registered name `procedural_predictor_plus_residual_correction_savings_v1` (currently cites placeholder `residual_hybrid_boosting_savings_v1` per FORMALIZATION_PENDING marker; empirically the registered equation exists)

### L1+ / CONSOLIDATION-OP / META (deferred; NOT blocking R2)

9. **META-CONSOLIDATE-OP-1**: extract `_pixel_shuffle_2x_nhwc` (correct convention from D=Z6) + general `_bilinear_resize_nhwc` (D=Z6's impl works for arbitrary target_h, target_w) to canonical `tac.local_acceleration.pr95_hnerv_mlx` so future MLX substrates inherit ONE source of truth per CLAUDE.md "consolidate into META layer" standing directive. Refactor A=DreamerV3 + D=Z6 + future Path 3 candidates to import from canonical.

10. **L1+ PERFORMANCE-OP** (D=Z6 advisory): `reconstruct_pair` O(max(pair_indices)) recurrence at 600 pairs is 599 predictor forwards per batch; consider rollout-then-gather optimization to reduce to O(max-unique-indices). NOT blocking R2.

11. **L1+ EQUATION-REGISTRY-OP** (D=Z6 advisory): register canonical Z6 equations (Rao-Ballard predictive coding update; Atick-Redlich cooperative-receiver mutual information; FiLM modulation per Perez 2018) to canonical equation registry per Catalog #344. Sister-substrate-wide gap; Phase 2 council symposium responsibility. NOT blocking R2.

---

## R2 readiness verdict

**R2 BLOCKED until FIX-WAVE-R1 lands the P0 op-routables:**

1. A-OP1 + A-OP2 (code fixes to A=DreamerV3 module.py)
2. E-OP1 + E-OP2 + E-OP3 (documentation fixes across 3 files)

After P0 ops land, verification:
- A=DreamerV3: re-run 11 tests; verify all PASS; re-measure `test_mlx_pytorch_decoder_parity_at_archive_boundary` max_abs drops below 5.0 (or 1.0 — best-case)
- E=BoostNeRV: re-run 25 tests; verify all PASS (no test changes expected since these are doc-only fixes)
- D=Z6: re-run 19 tests; verify all PASS (no changes — sister-coherence preserved)

If verification passes, R2 review subagent can fire per CLAUDE.md "Recursive adversarial review protocol — close paths" — taking different adversarial perspectives per protocol item 1 (rotation across the 4-co-lead structure + sister inner-council voices).

P1 + P2 op-routables (A-OP3, A-OP4, E-OP4) SHOULD land in the same FIX-WAVE-R1 commit batch but are NOT strict blockers for R2 (they are verification / amendment / advisory).

L1+ META-CONSOLIDATE-OPs (#9-11) are deferred to future subagents; do NOT block R2.

---

## Counter state per CLAUDE.md "Recursive adversarial review protocol — close paths"

- **Before R1**: counter = 0 (clean baseline)
- **Per-landing post-R1**:
  - A=DreamerV3: NOT CLEAN → counter resets to 0
  - D=Z6: CLEAN → counter would advance to 1/3 IF aggregated alone
  - E=BoostNeRV: NOT CLEAN → counter resets to 0
- **Aggregate post-R1**: counter = **0/3** (any NOT CLEAN resets the counter per protocol item 3)
- **Post-FIX-WAVE-R1 + R2-CLEAN**: counter advances to 1/3
- **Post-R3-CLEAN**: counter advances to 2/3
- **Post-R4-CLEAN**: counter advances to 3/3 → cycle closes per protocol gate
- **Operator-declared SEAL (D-1, conservative)**: alternative close path per "Recursive adversarial review protocol — close paths" amendment 2026-05-XX; requires (a) external-adversary unanimous SEAL + (b) Contrarian SUPER-VETO invoked + (c) 7-day cool-down + (d) operator explicit invoke. NOT applicable here (counter-advance path is straightforward; D-1 is for structurally-unsatisfiable cycles per R12-D meta-finding).

---

## Discipline applied

- **Catalog #229 PV**: 3 landing memos + 4+ source files per substrate read in full; 11+19+25 = 55 tests run before any review claim; canonical bilinear helper source + sister cross-comparison; canonical equation registry empirically queried
- **Catalog #110/#113 APPEND-ONLY**: 4 NEW memos (3 per-substrate + 1 aggregate); sister landing memos NEVER mutated
- **Catalog #117/#157/#174/#235/#289**: ALL commits via canonical `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256`
- **Catalog #119**: Co-Authored-By Claude trailer
- **Catalog #287**: every finding carries `[empirical:<measurement>]` evidence-tag; no placeholder rationales (every assumption-adversary verdict ≥4 chars; placeholder `<rationale>` / `<reason>` literals REJECTED)
- **Catalog #208**: docs/local-paths — only relative paths cited; no `/Users/adpena/` / `/tmp/` / `/home/` / Tailscale IPs / etc.
- **Catalog #292**: per-axis council member operating-within assumption surfaced explicitly in frontmatter (all 3 per-substrate memos + this aggregate)
- **Catalog #300 v2**: full frontmatter on all 4 memos (tier T2 per-substrate; T3 aggregate; attendees include the canonical 4-co-lead structure Shannon+Dykstra+Rudin+Daubechies per 2026-05-19 amendment + the relevant inner council voices; mission_contribution frontier_protecting; horizon_class frontier_pursuit)
- **Catalog #346**: canonical council roster `validate_council_dispatch_roster` returns complete=True for T3 aggregate (Shannon LEAD + Dykstra CO-LEAD + Rudin CO-LEAD + Daubechies CO-LEAD + Tao + Carmack + Hotz + Quantizr + MacKay + Selfcomp + Ballé + Hassabis + PR95Author + Contrarian + Assumption-Adversary = 15 attendees; ≥12-of-20 grand council quorum honored)
- **Catalog #340**: sister-checkpoint guard PROCEED before any edits; no overlap with the 5 in-flight sister subagents B'/C'/F/G/H per Catalog #230 ownership map
- **Catalog #206**: checkpoint discipline every ~10 tool uses
- **Catalog #126**: lane `lane_path_3_recursive_adversarial_review_r1_3_axis_landings_a_d_e_20260526` pre-registered
- **Catalog #294 9-dim checklist** (1-9): UNIQUENESS (the 3-axis cross-substrate aggregate review is the first of its kind for Path 3); BEAUTY + ELEGANCE (4 memos ≤700 lines each); DISTINCTNESS (NOT a sister review; this is an AGGREGATE synthesis with cross-substrate META findings); RIGOR (PV + empirical measurement + canonical equation verification); OPTIMIZATION PER TECHNIQUE (per-axis council members rotate per protocol); STACK-OF-STACKS-COMPOSABILITY (aggregate composes 3 per-substrate verdicts); DETERMINISTIC REPRODUCIBILITY (reproducer commands documented per Axis 2); EXTREME OPTIMIZATION + PERFORMANCE (R1 review takes ~3-4h wall-clock vs paid-dispatch cost $0); OPTIMAL MINIMAL CONTEST SCORE (R1 is QUALITY GATE not score-claim; non-promotable per Catalog #287 / #341 Tier A)
- **CLAUDE.md "Recursive adversarial review protocol — close paths"** items 1-8: this aggregate review IS R1; counter resets to 0; FIX-WAVE-R1 successor subagent required
- **CLAUDE.md "Council conduct" amendment 2026-05-19 4-co-lead structure**: T3 aggregate roster includes all 4 co-leads per Catalog #346 requirement
- **CLAUDE.md "Executing actions with care"**: review-only (NO code modifications); fixes are FIX-WAVE-R1 successor subagent's scope

---

## Cross-references

- **R1 per-substrate review memos**:
  - A=DreamerV3: `.omx/research/path_3_a_recursive_adversarial_review_r1_3_axis_20260526.md`
  - D=Z6: `.omx/research/path_3_d_recursive_adversarial_review_r1_3_axis_20260526.md`
  - E=BoostNeRV: `.omx/research/path_3_e_recursive_adversarial_review_r1_3_axis_20260526.md`
- **Landing memos under review**:
  - A=DreamerV3: `.omx/research/dreamer_v3_rssm_mlx_scaffold_landed_20260526.md` (commit `69253a1cc`)
  - D=Z6: `.omx/research/z6_predictive_coding_mlx_scaffold_landed_20260526.md` (commit `83b9ee3e2`)
  - E=BoostNeRV: `.omx/research/path_3_e_boost_nerv_against_pr110_substrate_design_20260526.md` (commit `83910e54e`)
- **Canonical references**:
  - Sister D=Z6 `_pixel_shuffle_2x_nhwc` (canonical correct convention): `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py::_pixel_shuffle_2x_nhwc`
  - Canonical bilinear helper: `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc`
  - Canonical equation registry: `tac.canonical_equations.query_equations()` (42 equations REGISTERED as of 2026-05-26)
  - Empirical anchor #1258 corrected: `.omx/research/pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526.md`
  - Empirical anchor #1265: `.omx/research/mlx_candidate_contest_equivalence_gate_landed_20260526.md`
- **In-flight sister subagents (DEFERRED per Catalog #230 ownership map; NOT reviewed in this R1)**:
  - B'=`ac4283983ece21b83` Z7-Mamba-2 cargo-cult-first 3-phase
  - C'=`ad26de7ad5f90848a` NSCS06 v8 chroma_lut cargo-cult-first 3-phase
  - F=`a23f0430835406351` Z8 hierarchical predictive coding
  - G=`ae952528954e27bef` NIRVANA cascading NeRV
  - H=`aba5069741fc4475b` ATW V2 cooperative-receiver cargo-cult-first
- **Lane**: `lane_path_3_recursive_adversarial_review_r1_3_axis_landings_a_d_e_20260526` L1 (impl_complete + memory_entry)

---

## Final aggregate verdict

**PROCEED_WITH_REVISIONS** — R1 NOT CLEAN; counter resets to 0; FIX-WAVE-R1 successor subagent required; R2 BLOCKED until P0 op-routables land and verification passes.

The substrate paradigms (categorical posterior DreamerV3 / FiLM-conditioned predictive coding / boosting-against-PR110-residual) are HARD-EARNED at the math + scientific + engineering level per Axis 1 across all 3 substrates. The IMPLEMENTATION-LEVEL gaps (2 in A=DreamerV3; 1 documentation-only in E=BoostNeRV) are TIGHTLY SCOPED and resolvable in a single FIX-WAVE-R1 commit batch (≤6 files; ≤200 LOC of edits).

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #307 paradigm-vs-implementation classification: these are IMPLEMENTATION-LEVEL findings requiring FIX-WAVE-R1, not paradigm-level kills. All 3 substrates remain `research_only=true` per their landing-time posture; the L0→L1 promotion path is unblocked by the FIX-WAVE-R1 + R2-R4 clean-pass cycle.

Estimated FIX-WAVE-R1 wall-clock: ~45-60min for a successor subagent with PV + canonical-serializer discipline. Estimated R2-R4 cycle: ~3-4h per round × 3 rounds = 9-12h total to reach SEAL via counter-advance path.

**Mission alignment per Catalog #300**: `frontier_protecting` — the R1 review prevents L0→L1 promotion of substrates with TRAINING-INVALIDATING MLX↔PyTorch drift bugs (which would silently corrupt L1+ score-aware-loss training and produce phantom-score Modal dispatches per Catalog #313 + #341 sister discipline). Closing FIX-WAVE-R1 + R2-R4 unblocks the canonical Path 3 substrate-class-shift pursuit at $0 cost.
