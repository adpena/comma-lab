<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE — this memo is the
     canonical FIX-WAVE-R1 closure landing for Path 3 R1 review findings
     against candidates A (DreamerV3 RSSM) + E (BoostNeRV against PR110).
     DO NOT mutate after landing; corrections must go in an APPEND-ONLY
     footer per Catalog #110 sister discipline. -->
<!-- Catalog #344 canonical equation cross-ref: FIX-WAVE-R1 is meta-discipline
     (close findings before R2 fires) NOT a new score-claim. No canonical
     equation declared at FIX-WAVE-R1 surface; the parent landings retain
     their canonical equation refs unchanged. -->
---
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Carmack
  - Hotz
  - Quantizr
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "FIX-WAVE-R1 can close A=DreamerV3 + E=BoostNeRV findings in a single commit batch"
    classification: HARD-EARNED
    rationale: "All R1 findings were TIGHTLY SCOPED per the aggregate review's pre-flight: 2 source-code edits to A=DreamerV3 module.py (A-OP1 + A-OP2) + 1 source-code docstring edit to A test (A-OP3) + 2 source-code docstring edits to E (E-OP2 + E-OP3) + 2 APPEND-ONLY design-memo footers (A-OP4 + E-OP1 + E-OP4). Total touch surface: 6 files; ≤300 LOC of edits. Sister-checkpoint guard PROCEED via tools/check_sister_checkpoint_before_git_add.py with zero overlap against 2 in-flight sister subagent's files_touched within 60-min lookback window."
  - assumption: "The post-fix max_abs=0.0054 satisfies the R1 review's stated < 5.0 (or best-case < 1.0) promotion criterion"
    classification: HARD-EARNED
    rationale: "Empirically measured immediately after A-OP1 + A-OP2 land (no further iteration required): max_abs=0.0054, mean_abs=0.0007 (~4500x improvement vs pre-fix 24.34 baseline). This is ~185x below the R1 review's stated < 1.0 best-case criterion and approaches fp32 compound-op precision noise floor across 6 PixelShuffle blocks + sin/sigmoid nonlinearities + final RGB heads. Threshold tightened from < 50.0 to < 0.05 (~10x headroom above empirical) per Catalog #287 evidence-tag discipline."
  - assumption: "The CONSOLIDATE-OP for L1+ (extract _pixel_shuffle_2x_nhwc + general _bilinear_resize_nhwc to canonical tac.local_acceleration.pr95_hnerv_mlx) is correctly DEFERRED from FIX-WAVE-R1 scope"
    classification: HARD-EARNED
    rationale: "Per the explicit charter instruction 'do NOT execute in FIX-WAVE-R1 scope (you focus on closing P0+P1+P2 findings; CONSOLIDATE is a separate canonical-helper promotion lane)' and per CLAUDE.md 'Subagent coherence-by-default' standing directive: the CONSOLIDATE-OP is an L1+ canonical-helper-promotion lane that requires its own per-substrate test suite + refactor pass across A=DreamerV3 + D=Z6 + future Path 3 candidates. Queued as TaskCreate op-routable; deferral does NOT block R2 readiness (the in-line canonical-helper IMPORT in A=DreamerV3's _bilinear_resize_2x_nhwc + the verbatim correct convention COPY in A=DreamerV3's _pixel_shuffle_2x_nhwc are functionally equivalent to the CONSOLIDATE-OP outcome at the per-substrate observable surface)."
council_decisions_recorded:
  - "FIX-WAVE-R1 CLOSED: 5 P0/P1/P2 op-routables from R1 review (A-OP1, A-OP2, A-OP3, A-OP4, E-OP1, E-OP2, E-OP3, E-OP4) all landed in a single commit batch"
  - "R2 readiness verdict: CLEAN for A=DreamerV3 + E=BoostNeRV; R2 can fire on a successor subagent's schedule"
  - "Counter advance: per CLAUDE.md 'Recursive adversarial review protocol — close paths' item 4, R2 unblocked; counter will advance to 1/3 IF R2 returns CLEAN"
  - "CONSOLIDATE-OP DEFERRED to L1+ per charter scope discipline; queued as TaskCreate op-routable for future canonical-helper-promotion subagent"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
canonical_equation_refs: []
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - path_3_recursive_adversarial_review_r1_aggregate_3_axis_landings_a_d_e_20260526
  - path_3_a_recursive_adversarial_review_r1_3_axis_20260526
  - path_3_d_recursive_adversarial_review_r1_3_axis_20260526
  - path_3_e_recursive_adversarial_review_r1_3_axis_20260526
  - dreamer_v3_rssm_mlx_scaffold_landed_20260526
  - path_3_e_boost_nerv_against_pr110_substrate_design_20260526
---

# Path 3 FIX-WAVE-R1: Close R1 review findings — LANDED 2026-05-26

**Lane**: `lane_path_3_fix_wave_r1_close_findings_20260526` L1 (impl_complete + memory_entry)
**Evidence grade**: `[macOS-MLX research-signal]` per CLAUDE.md "MLX portable-local-substrate authority"
**Cost**: $0 + ~30 min wall-clock (6 file edits + 11+25 test re-runs + landing memo); NO paid CUDA dispatch.

## Verdict

**FIX-WAVE-R1 CLOSED** — all 5 P0/P1/P2 op-routables landed; R2 can fire.

Per CLAUDE.md "Recursive adversarial review protocol — close paths" item 2 + item 4: FIX-WAVE-R1 successor subagent landed all R1 issues + committed BEFORE R2 begins. R1 counter state remains 0/3 (R1 did NOT advance the counter because A + E both returned PROCEED_WITH_REVISIONS); R2 is now unblocked and a successor R2 subagent can fire with clean preconditions.

## Per-finding closure verdict

| Op-routable | Priority | Surface | Closure verdict | Evidence |
|---|---|---|---|---|
| **A-OP1** | P0 / CRITICAL | `src/tac/substrates/dreamer_v3_rssm/module.py::_pixel_shuffle_2x_nhwc` | **CLOSED** (in-place source edit) | Channel-FIRST reshape `(B, H, W, out_C, 2, 2)` + transpose `(0, 1, 4, 2, 5, 3)` matching sister D=Z6 + canonical PR95 helper convention. Empirically PyTorch-byte-stable. |
| **A-OP2** | P0 / CRITICAL | `src/tac/substrates/dreamer_v3_rssm/module.py::_bilinear_resize_2x_nhwc` | **CLOSED** (in-place source edit) | Delegates to canonical `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc`. Catalog #295 self-containment preserved (inflate.py is PyTorch-only; canonical helper imported at MLX training time only). |
| **A-OP3** | P1 / VERIFICATION | `src/tac/substrates/dreamer_v3_rssm/tests/test_basic.py::test_mlx_pytorch_decoder_parity_at_archive_boundary` | **CLOSED** (in-place test edit) | Threshold tightened from `< 50.0` to `< 0.05`; empirical post-fix max_abs=0.0054, mean_abs=0.0007 (~4500x improvement vs pre-fix 24.34; ~185x below R1's best-case < 1.0). |
| **A-OP4** | P2 / DOCUMENTATION | `.omx/research/dreamer_v3_rssm_mlx_scaffold_landed_20260526.md` | **CLOSED** (APPEND-ONLY footer per Catalog #110/#113) | Cargo-cult audit row #5 amendment recorded in APPEND-ONLY footer; body preserved UNMUTATED; correction documents the implementation-level distinction between conv-layout transpose correctness (HARD-EARNED) vs full decoder forward equivalence (required A-OP1 + A-OP2 fixes). |
| **E-OP1** | P0 / DOCUMENTATION | `.omx/research/path_3_e_boost_nerv_against_pr110_substrate_design_20260526.md` | **CLOSED** (APPEND-ONLY footer per Catalog #110/#113) | BPR1 header byte count corrections recorded in APPEND-ONLY footer (3 surfaces: §"Canonical-vs-unique" line 67 + §"9-dim" line 84 + §"Byte accounting" line 197); body preserved UNMUTATED; canonical truth is `BPR1_HEADER_LEN = 29` per source constant. |
| **E-OP2** | P0 / DOCUMENTATION | `src/tac/substrates/boost_nerv_pr110_residual/archive.py` module docstring | **CLOSED** (in-place source-code docstring edit) | Line 8 corrected from "BPR1 header 28 bytes" to "BPR1 header 29 bytes". Source-code docstrings are in-place editable per source-code evolution discipline. |
| **E-OP3** | P0 / DOCUMENTATION | `src/tac/substrates/boost_nerv_pr110_residual/__init__.py` archive grammar comment | **CLOSED** (in-place source-code comment edit) | Lines 40-47 corrected from "24-byte header" to "29-byte header" + expanded to include the `align[1]` field that the original mental model omitted. |
| **E-OP4** | P2 / ADVISORY | `.omx/research/path_3_e_boost_nerv_against_pr110_substrate_design_20260526.md` line 234 | **CLOSED** (APPEND-ONLY footer per Catalog #110/#113) | Canonical equation name correction recorded in APPEND-ONLY footer: registered name is `procedural_predictor_plus_residual_correction_savings_v1` (REGISTERED in `tac.canonical_equations` registry as of 2026-05-26). |

**No findings classified RESIDUAL-DRIFT-for-R2**: all closures CLEAN; R2 can fire without blocker.

## Post-fix test verdict (A=DreamerV3 decoder parity measurements)

```
$ .venv/bin/python -m pytest src/tac/substrates/dreamer_v3_rssm/tests/test_basic.py::test_mlx_pytorch_decoder_parity_at_archive_boundary -v -s
MLX↔PyTorch decoder parity: max_abs=0.0054, mean_abs=0.0007
PASSED
```

| Metric | Pre-FIX-WAVE-R1 (R1 measurement) | Post-FIX-WAVE-R1 (this commit batch) | Improvement |
|---|---|---|---|
| max_abs (range [0, 255]) | 24.34 | **0.0054** | ~4500x |
| mean_abs (range [0, 255]) | not measured | **0.0007** | new measurement |
| Test threshold | `< 50.0` (loose; documents bug honestly per Catalog #287) | `< 0.05` (~10x headroom above empirical) | tightened ~1000x |
| R1's stated < 5.0 promotion criterion | NOT MET (24.34 > 5.0) | **MET** (0.0054 ≪ 5.0); ~925x headroom | criterion exceeded |
| R1's stated best-case < 1.0 | NOT MET (24.34 ≫ 1.0) | **MET** (0.0054 ≪ 1.0); ~185x headroom | criterion exceeded |

## Post-fix test verdict (full suite, no regressions)

```
$ .venv/bin/python -m pytest src/tac/substrates/dreamer_v3_rssm/tests/test_basic.py -v
============================== 11 passed in 0.71s ==============================

$ .venv/bin/python -m pytest src/tac/substrates/boost_nerv_pr110_residual/tests/ -v
============================== 25 passed in 0.17s ==============================
```

11 + 25 = 36/36 tests pass; no regressions; all closures structurally clean.

## R2 readiness signal

**CLEAN → R2 CAN FIRE.**

Per CLAUDE.md "Recursive adversarial review protocol — close paths" items 1-8: FIX-WAVE-R1 successor subagent has landed all R1 issues; the counter remains 0/3 pending R2 verdict; R2 successor subagent is unblocked.

Recommended R2 scope:
- Re-verify A=DreamerV3 + E=BoostNeRV + D=Z6 (D=Z6 passed CLEAN at R1; expected to remain CLEAN at R2 unless new bug surfaces).
- Rotate adversarial perspectives per protocol item 1 (different inner council voices than R1; e.g. R2 could foreground Rudin + Daubechies + Mallat + Schmidhuber if Axis 1 is the focal axis; or rotate to Tao + Boyd + Ballé if Axis 2 MLX drift is focal).
- Per protocol item 8 (NEW assumption-challenge axis): R2 MUST explicitly answer "What shared assumption is this work operating within, and would violating it unlock breakthrough?" The most-recent META-ASSUMPTION review's classification of the 3-axis methodology itself as HARD-EARNED-vs-CARGO-CULTED is a candidate question for R2's Assumption-Adversary seat.

## CONSOLIDATE-OP queued for L1+ (NOT in FIX-WAVE-R1 scope)

Per the charter's explicit deferral instruction + CLAUDE.md "consolidate everything into META layer" standing directive: the META-CONSOLIDATE-OP-1 (extract `_pixel_shuffle_2x_nhwc` + general `_bilinear_resize_nhwc` to canonical `tac.local_acceleration.pr95_hnerv_mlx`) is queued as TaskCreate op-routable for a future canonical-helper-promotion subagent. This is L1+ canonical-helper-promotion work requiring:

1. New canonical helper function signatures + dedicated unit tests at `src/tac/local_acceleration/pr95_hnerv_mlx.py` tests.
2. Per-substrate refactor: A=DreamerV3, D=Z6, future Path 3 candidates B'/C'/F/G/H/K (8 substrates total).
3. Sister Catalog #335 cathedral-consumer canonical-contract regression audit.
4. New STRICT preflight gate (sister of Catalog #335) refusing locally-invented MLX primitives at substrate scope (if the operator approves the META-gate landing).

This is bounded ~2-3h wall-clock work; deferred per charter scope discipline. NOT a blocker for R2.

## 6-hook wire-in declaration (per Catalog #125)

- **hook #1 sensitivity-map**: N/A (FIX-WAVE-R1 is meta-discipline; no signal contribution).
- **hook #2 Pareto constraint**: N/A.
- **hook #3 bit-allocator**: N/A.
- **hook #4 cathedral autopilot dispatch**: N/A (FIX-WAVE-R1 unblocks future autopilot consumption of A=DreamerV3 + E=BoostNeRV; THIS landing itself does not directly produce signal for the ranker).
- **hook #5 continual-learning posterior**: ACTIVE (R2 successor subagent can append PROCEED-or-PROCEED_WITH_REVISIONS verdict per Catalog #300 v2 frontmatter once R2 fires; council deliberation posterior at `.omx/state/council_deliberation_posterior.jsonl` will accumulate the R2 verdict as a chronologically-later anchor superseding R1's PROCEED_WITH_REVISIONS).
- **hook #6 probe-disambiguator**: N/A (the canonical contest-equivalence gate at Catalog #1265 IS the disambiguator; THIS FIX-WAVE-R1 unblocks A=DreamerV3 from satisfying that gate's `|S_MLX - S_PT| ≤ 0.001 contest-units` threshold by closing the decoder forward semantics drift).

## Reactivation criteria (per CLAUDE.md "Forbidden premature KILL")

This landing has no kill verdicts; it CLOSES findings rather than retiring lanes. Both A=DreamerV3 + E=BoostNeRV remain `research_only=true` per their respective landing-time posture; their L0→L1 promotion path is now unblocked by R2-R4 clean-pass cycle.

Reactivation criteria for R2 successor subagent:
1. Sister-checkpoint guard PROCEED before any edits (`tools/check_sister_checkpoint_before_git_add.py`).
2. Re-read all 3 R1 per-substrate review memos + this FIX-WAVE-R1 landing + the 2 APPEND-ONLY footers BEFORE drafting R2 verdict.
3. Re-run 11+25+19 = 55 tests; verify all PASS.
4. Rotate adversarial perspectives per protocol item 1.
5. Per protocol item 8, answer the assumption-challenge question explicitly.

## Discipline applied

- **Catalog #229 PV**: 7 source/test files + 3 R1 review memos + 1 aggregate review memo + 2 design memos + canonical bilinear helper source + sister D=Z6 PixelShuffle source all read in full BEFORE any edit.
- **Catalog #117 / #157 / #174 / #235 / #289**: ALL commits via canonical `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` for every file (this landing's commit will use the canonical serializer).
- **Catalog #119**: Co-Authored-By Claude trailer.
- **Catalog #206**: checkpoint discipline (3 checkpoints emitted to `.omx/state/subagent_progress.jsonl`).
- **Catalog #208**: docs/local-paths — only relative paths cited; no `/Users/adpena/` / `/tmp/` / `/home/` / Tailscale IPs.
- **Catalog #230**: sister-subagent ownership map (zero overlap with 2 in-flight sister subagents per the explicit charter ownership map).
- **Catalog #287**: every finding closure carries `[empirical:<measurement>]` evidence-tag; no placeholder rationales.
- **Catalog #295**: submission inflate self-containment preserved (A-OP2 canonical helper imported at MLX training time only; inflate.py remains PyTorch-only).
- **Catalog #300 v2**: full frontmatter on this landing memo + the 2 APPEND-ONLY footers (tier T2; mission_contribution frontier_protecting; horizon_class frontier_pursuit).
- **Catalog #340**: sister-checkpoint guard PROCEED via `tools/check_sister_checkpoint_before_git_add.py` (output: "OK: PROCEED: caller's 6 non-exempt file(s) do not overlap any of 2 in-flight sister subagent's files_touched within the 60-minute lookback window").
- **Catalog #110/#113 APPEND-ONLY**: design memo corrections via APPEND-ONLY FOOTERS (not in-place mutation); source-code docstrings ARE in-place editable per source-code evolution discipline.
- **CLAUDE.md "Recursive adversarial review protocol — close paths"** items 1-8: FIX-WAVE-R1 successor subagent landed all R1 issues before R2 fires per item 4.
- **CLAUDE.md "Executing actions with care"**: NO `gh pr create`, NO Modal/Vast/Lightning dispatch (all closures are local source-code + memo edits; no paid GPU spend).

## Sister coordination (Catalog #230)

Per charter ownership map at landing time:
- **LANDED** sisters (not touched in this scope): A=DreamerV3 `69253a1cc`, B'=Z7-Mamba-2-v2 `7a103fdbb`, C'=NSCS06-v8 `f59c8401b`, D=Z6 `83b9ee3e2`, E=BoostNeRV `83910e54e`, R1 review `80acd6da3`, G=NIRVANA `f7d2e86fe`. Scope: only `module.py` + 2 docstrings + 2 APPEND-ONLY footers (within A/E scope).
- **IN-FLIGHT** sisters (not touched per Catalog #230): F=Z8 hierarchical predictive coding, H=ATW V2, K=COIN++.
- Sister-checkpoint guard verified zero file overlap before any edit.

## Cost

- **Cost**: $0 (no paid GPU dispatch).
- **Wall-clock**: ~30 min (PV 5 min + 6 file edits 10 min + 36 tests 2 min + landing memo + commit 13 min).
- **Modal / Lightning / Vast.ai**: NOT invoked.
- **Codex**: NOT invoked (operator-routable per CLAUDE.md "Pre-dispatch codex review automation" Catalog #271 — N/A for documentation+test+source-fix scope).

## Artifact paths

- A=DreamerV3 source fix: `src/tac/substrates/dreamer_v3_rssm/module.py` lines 184-243 (post-edit)
- A=DreamerV3 test threshold tightening: `src/tac/substrates/dreamer_v3_rssm/tests/test_basic.py` lines 320-350 (post-edit)
- A=DreamerV3 landing memo APPEND-ONLY footer: `.omx/research/dreamer_v3_rssm_mlx_scaffold_landed_20260526.md` lines 307+ (post-edit)
- E=BoostNeRV archive.py docstring fix: `src/tac/substrates/boost_nerv_pr110_residual/archive.py` lines 1-26 (post-edit)
- E=BoostNeRV __init__.py comment fix: `src/tac/substrates/boost_nerv_pr110_residual/__init__.py` lines 40-47 (post-edit)
- E=BoostNeRV design memo APPEND-ONLY footer: `.omx/research/path_3_e_boost_nerv_against_pr110_substrate_design_20260526.md` lines 235+ (post-edit)
- This landing memo: `.omx/research/path_3_fix_wave_r1_close_findings_landed_20260526.md`
- Lane registry: pending L1 row registration via `tools/lane_maturity.py` for `lane_path_3_fix_wave_r1_close_findings_20260526`

## Reproduce

```bash
# Verify all post-fix tests pass (11 + 25 = 36)
.venv/bin/python -m pytest src/tac/substrates/dreamer_v3_rssm/tests/test_basic.py -v
.venv/bin/python -m pytest src/tac/substrates/boost_nerv_pr110_residual/tests/ -v

# Re-measure A=DreamerV3 decoder parity (expect max_abs ≈ 0.005, mean ≈ 0.0007)
.venv/bin/python -m pytest src/tac/substrates/dreamer_v3_rssm/tests/test_basic.py::test_mlx_pytorch_decoder_parity_at_archive_boundary -v -s

# Verify BPR1 header struct size (expect 29)
.venv/bin/python -c "import struct; print(struct.calcsize('<5sBBB16sIB'))"

# Verify source-code docstrings now consistent
grep -n "29-byte\|29 bytes" src/tac/substrates/boost_nerv_pr110_residual/{archive.py,__init__.py}
```

## Cross-references

- R1 aggregate review (parent): `.omx/research/path_3_recursive_adversarial_review_r1_aggregate_3_axis_landings_a_d_e_20260526.md`
- R1 per-substrate reviews:
  - A=DreamerV3: `.omx/research/path_3_a_recursive_adversarial_review_r1_3_axis_20260526.md`
  - D=Z6: `.omx/research/path_3_d_recursive_adversarial_review_r1_3_axis_20260526.md` (CLEAN; not in FIX-WAVE-R1 scope)
  - E=BoostNeRV: `.omx/research/path_3_e_recursive_adversarial_review_r1_3_axis_20260526.md`
- Landing memos under correction:
  - A=DreamerV3: `.omx/research/dreamer_v3_rssm_mlx_scaffold_landed_20260526.md` (APPEND-ONLY footer added)
  - E=BoostNeRV: `.omx/research/path_3_e_boost_nerv_against_pr110_substrate_design_20260526.md` (APPEND-ONLY footer added)
- Canonical references:
  - Sister D=Z6 `_pixel_shuffle_2x_nhwc`: `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py` lines 361-372
  - Canonical PR95 helper: `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc` + `pixel_shuffle_2x_nhwc`
  - Canonical equation registry: `tac.canonical_equations.query_equations()` (REGISTERED `procedural_predictor_plus_residual_correction_savings_v1` as of 2026-05-26)
- Operator pacing directive 2026-05-26: "Keep feeding the queue but we need to be mindful not to outpace session rate limits" → this landing minimized tool calls (combined related reads/edits; ~30 min wall-clock per the bounded scope estimate).

## Final verdict

**FIX-WAVE-R1 CLOSED CLEAN — R2 CAN FIRE.**

All 5 P0/P1/P2 op-routables (A-OP1 + A-OP2 + A-OP3 + A-OP4 + E-OP1 + E-OP2 + E-OP3 + E-OP4) landed in a single commit batch per CLAUDE.md "Recursive adversarial review protocol — close paths" item 2. R2 successor subagent is unblocked; expected R2 verdict is CLEAN (no residual drift expected; the bug fixes are structurally complete per the empirical drift measurement collapsing from 24.34 → 0.0054).

CONSOLIDATE-OP for L1+ canonical-helper promotion is queued as TaskCreate op-routable per charter scope discipline; NOT in FIX-WAVE-R1 scope.

Per CLAUDE.md "Mission alignment per Catalog #300": `frontier_protecting` — the FIX-WAVE-R1 closure prevents L0→L1 promotion of A=DreamerV3 with TRAINING-INVALIDATING MLX↔PyTorch drift bugs (which would silently corrupt L1+ score-aware-loss training and produce phantom-score Modal dispatches per Catalog #313 + #341 sister discipline). Closing FIX-WAVE-R1 + R2-R4 unblocks the canonical Path 3 substrate-class-shift pursuit at $0 cost.
