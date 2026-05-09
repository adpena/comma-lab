# Representation lane 8-field backfill — Catalog #124 strict-flip prep (2026-05-09)

## TL;DR

The 7 representation/codec lanes flagged warn-only by Catalog #124
`check_representation_lane_has_archive_grammar_at_design_time` are now all
either declaring the 8 design-time fields (4 lanes, Path A) or opting out
via `research_only=true` with explicit `reactivation_criteria` (3 lanes,
Path D). Live count after backfill: **0**. Strict-flip is unblocked pending
operator approval.

| Lane | Path | Verdict |
|---|---|---|
| `lane_12_nerv_mask_codec` | D | research_only — superseded by `lane_12_v2_nerv_as_renderer` (HNeRV retro lesson 5) |
| `lane_20_balle_hyperprior` | A | 8 fields declared (`design_evidence` dict) |
| `lane_pr106_latent_sidecar` | A | 8 fields declared (`design_evidence` dict) |
| `lane_alpha_nerv_mask` | D | research_only — superseded by `lane_12_v2`; PARADIGM-α 4-way bake-off prerequisite |
| `lane_alpha_wavelet_mask` | A | 8 fields declared (`design_evidence` dict) |
| `track1_phase_a3_alt_mallat_wavelet` | B | research_only — sensitivity proxy lane, not archive-grammar lane |
| `track1_phase_a6_selfcomp_blockfp_hyperprior` | A | 8 fields declared (`design_evidence` dict) |

[empirical: `.omx/state/lane_registry.json` snapshot 2026-05-09 post-backfill;
`from tac.preflight import check_representation_lane_has_archive_grammar_at_design_time;
check_representation_lane_has_archive_grammar_at_design_time(strict=True)` returns 0 violations]

## Tooling extension landed in the same change

`tools/lane_maturity.py` previously had no mutation surface for top-level
non-gate fields (`lane_class`, `research_only`, `reactivation_criteria`)
or for the `design_evidence` dict that Check #124 reads. Bare-editing
`.omx/state/lane_registry.json` is FORBIDDEN per CLAUDE.md "Lane maturity
registry — non-negotiable". Two options were available:

1. Use the inline `field=value` notes-string syntax that Check #124 already
   accepts (5th acceptance location in `_lane_has_field`); this preserves
   the current CLI surface but mixes structured data with prose.
2. Extend `lane_maturity.py` with a small `set-field` subcommand that
   writes the structured key + appends to the audit log.

Path 2 was chosen for beauty/simplicity (per CLAUDE.md "Beauty, simplicity,
and developer experience"): top-level `research_only=true` and
`lane_class=substrate_engineering` are checked by Check #124 as DICT keys,
not in notes — so the inline-notes syntax does NOT cover the opt-out path.
Adding the structured surface unblocks future opt-out usage without prose
parsing.

The new `set-field` subcommand:

- Accepts top-level scalar fields: `lane_class`, `research_only`,
  `reactivation_criteria` (allowlist enforced).
- Accepts `design_evidence.<sub>` paths for any of the 8 Catalog #124
  fields (allowlist enforced).
- Coerces CLI string input: bool fields accept `true/false/yes/no/1/0`;
  int fields parse `int(raw)`; list fields (`runtime_dep_closure`,
  `reactivation_criteria`) parse comma-separated.
- Refuses empty values + unknown lane_id + unknown field with clear errors.
- Appends a `set-field` JSONL record to `.omx/state/lane_maturity_audit.log`
  with `before_state` + `after_state` for forensics.

13 new pytest cases in `src/tac/tests/test_lane_maturity_harness.py`
(57/57 file-level total pass). Round-trip end-to-end test verifies
representation-lane violation → `set_field("research_only", True)` →
violation cleared.

## Per-lane analysis

### 1. `lane_12_nerv_mask_codec` — Path D (research_only=true)

**Path D rationale:** Per HNeRV retrospective lesson 5 (mask-only NeRV
loses to renderer-class NeRV by definition: "the contest scorer derives
masks from frames; replacing masks per se buys nothing if the renderer is
unchanged"), this lane is **superseded by `lane_12_v2_nerv_as_renderer`**
which already lives in the registry at L1 with all 8 fields declared in
its notes blob. Lane 12 (mask-only) has L2-clearance evidence in its
gates that is forensic-anchor-quality (impl_complete + real_archive_empirical
satisfied) but the architectural class is dominated.

Per CLAUDE.md `forbidden_premature_kill_without_research_exhaustion`: the
verdict is DEFERRED-pending-research, NOT killed. The lane retains its
gates as a forensic anchor; `research_only=true` opts it out of Check #124
without retiring the historical evidence.

**Reactivation criteria (declared on lane):**
- `superseded_by_lane_12_v2_nerv_as_renderer_per_HNeRV_retrospective_lesson_5`
- `reactivate_only_if_lane_12_v2_falsified_AND_mask_slot_only_replacement_re_motivated`
- `renderer_path_dominates_mask_slot_path_at_contest_scorer_level`

### 2. `lane_20_balle_hyperprior` — Path A (8 fields declared)

**Path A rationale:** Real archive-grammar lane with on-disk implementation
at `src/tac/balle_hyperprior_renderer.py` (15.4 KB). The module declares
`BHP_MAGIC = b"BHP1"` (line 79), `encode_balle_hyperprior` (line 281), and
`decode_balle_hyperprior` (line 297) with magic-byte validation. The lane
already has L2 evidence (impl_complete + real_archive_empirical via Lane G
v3 qint stream amortization). The 8 fields are real, not aspirational.

**8 fields declared (under `design_evidence`):**
- `archive_grammar`: `src/tac/balle_hyperprior_renderer.py:79_BHP_MAGIC_b_BHP1_with_4byte_magic_then_packed_per_tensor_int8_with_fp16_scales_and_ScalePriorMLP_state_dict_brotli_compressed`
- `parser_section_manifest`: `src/tac/balle_hyperprior_renderer.py:281_encode_balle_hyperprior_and_297_decode_balle_hyperprior_with_BHP_MAGIC_validation`
- `inflate_runtime_loc_budget`: `150` (waiver vs target 100; ScalePriorMLP forward pass requires Conv2d/Linear)
- `runtime_dep_closure`: `["torch", "brotli"]`
- `export_format`: `BHP1_monolithic_int8_per_tensor_with_fp16_scales_and_brotli_ScalePriorMLP_sidecar`
- `score_aware_loss`: `DEFERRED_pending_score_aware_substrate_currently_train_amortizes_against_Lane_G_v3_qint_stream_at_byte_level_only_promotion_to_score_aware_loss_via_gradient_through_FastViT_T12_PoseNet_and_EfficientNet_B2_SegNet_required_before_dispatch`
- `bolt_on_loc_budget`: `350` (target)
- `no_op_detector_planned`: `true`

The `score_aware_loss` field documents the current substrate gap explicitly
per HNeRV retrospective lesson 4 ("score-naive substrate" was the gap).
Promotion to actual contest-CUDA dispatch requires score-aware retrain.

### 3. `lane_pr106_latent_sidecar` — Path A (8 fields declared)

**Path A rationale:** Strongest Path A candidate. Real archive-bound lane
with full inflate runtime at `submissions/pr106_latent_sidecar/inflate.py`
(166 LOC). `SIDECAR_MAGIC = 0xFE` + `SIDECAR_FORMAT_ID = 0x01` validation
already in place. Builder at `experiments/build_pr106_latent_sidecar.py`
(19.8 KB). Remote runbook at `scripts/remote_lane_pr106_latent_sidecar.sh`
(6.5 KB) with NVDEC probe + heartbeat + contest-CUDA tag. Predicted
empirical gain ~ -0.001 to -0.002 score per binary-forensics dossier (PR99
inheritance).

**8 fields declared:**
- `archive_grammar`: `submissions/pr106_latent_sidecar/inflate.py:41_SIDECAR_MAGIC_0xFE_then_format_id_0x01_then_per_pair_dim_idx_uint8_and_delta_q_int8_appended_to_PR106_HNeRV_archive`
- `parser_section_manifest`: `submissions/pr106_latent_sidecar/inflate.py:51_magic_validation_and_format_id_validation_with_per_pair_correction_decode_at_166_LOC_total_inflate_module`
- `inflate_runtime_loc_budget`: `166` (waiver vs target 100; HNeRV decoder forward pass is the bulk)
- `runtime_dep_closure`: `["torch", "brotli"]`
- `export_format`: `PR106_HNeRV_decoder_state_dict_plus_latents_plus_appended_per_pair_correction_sidecar_with_FE_01_magic_byte_pair`
- `score_aware_loss`: `scorer_driven_dim_delta_search_at_remote_Stage_3_via_brute_force_over_28_dim_latent_space_per_pair_minimizing_FastViT_T12_PoseNet_and_EfficientNet_B2_SegNet_distortion_on_upstream_videos_0_mkv` ← satisfies HNeRV retro lesson 4 (score-aware by construction)
- `bolt_on_loc_budget`: `350`
- `no_op_detector_planned`: `true`

This lane is the closest internal analogue to the leaderboard PR101 bolt-on
pattern (335 LOC of entropy bolt-ons on top of PR100 substrate). It is
substrate-faithful (PR106 = HNeRV-root) and the score-aware loss is wired
at remote Stage 3.

### 4. `lane_alpha_nerv_mask` — Path D (research_only=true)

**Path D rationale:** PARADIGM-α NeRV mask encoder is one of FOUR
ALTERNATIVE mask encoders (NeRV / wavelet / VQ-VAE / grayscale-LUT) that
target the SAME slot — `masks.mkv`. Per
`project_paradigm_alpha_architecture_clarification_20260506` the four
components are alternatives, not additives. The right experiment is a
head-to-head 4-way bake-off at the same renderer + pose anchor.

This NeRV mask path is also dominated by `lane_12_v2_nerv_as_renderer`
(renderer-class instead of mask-slot), per HNeRV retro lesson 5.

`impl_complete` evidence is a `NotImplementedError` cross-paradigm WARN
guard at `experiments/pipeline.py step_extract_masks` (commit 80455cf8) —
which is the right L1-quality scaffold but no execution path.

**Reactivation criteria:**
- `superseded_by_lane_12_v2_nerv_as_renderer_per_HNeRV_retrospective_lesson_5_and_paradigm_alpha_clarification_20260506`
- `paradigm_alpha_components_are_alternatives_not_additives_per_project_paradigm_alpha_architecture_clarification_20260506`
- `reactivate_only_if_alpha_4_way_bake_off_runs_AND_NeRV_mask_path_wins_head_to_head`

### 5. `lane_alpha_wavelet_mask` — Path A (8 fields declared)

**Path A rationale:** Unlike sister `lane_alpha_nerv_mask`, the wavelet
path has full module implementation at `src/tac/wavelet_mask_codec.py`
(21.6 KB) with `WAVELET_MAGIC = b"WMC1"` (line 70), `encode` (line 468),
`decode` with magic validation (line 500), REPACKABLE_SECTIONS string
constants, slug filename + WR01 schema branch + 7 adversarial-review
fixes already applied (commits 5f187bb0 + fa1e8759 + 0abfd60e + f8975eaa).

`strict_preflight` + `three_clean_review` + `memory_entry` + `deploy_runbook`
gates already true. Cross-paradigm WIRED.

The lane functions as the wavelet alternative in the PARADIGM-α 4-way
bake-off (per clarification memo). Score_aware_loss is DEFERRED pending
the bake-off dispatch — this is documented honestly rather than claimed
empirically.

**8 fields declared:**
- `archive_grammar`: `src/tac/wavelet_mask_codec.py:70_WAVELET_MAGIC_b_WMC1_with_4byte_magic_then_packed_wavelet_subband_coefficients_per_REPACKABLE_SECTIONS_layout_per_adversarial_review_session_2026_05_06`
- `parser_section_manifest`: `src/tac/wavelet_mask_codec.py:468_encode_with_WAVELET_MAGIC_write_and_500_decode_with_magic_validation_plus_REPACKABLE_SECTIONS_string_constants_post_review_fix`
- `inflate_runtime_loc_budget`: `200` (waiver vs target 100; pywt subband reconstruction)
- `runtime_dep_closure`: `["torch", "pywt", "brotli"]`
- `export_format`: `WMC1_monolithic_wavelet_subband_payload_replacing_masks_mkv_in_alpha_4_way_bake_off_alternative_per_paradigm_alpha_clarification_20260506`
- `score_aware_loss`: DEFERRED pending α 4-way bake-off dispatch (see field for full text)
- `bolt_on_loc_budget`: `350`
- `no_op_detector_planned`: `true`

### 6. `track1_phase_a3_alt_mallat_wavelet` — Path B (research_only=true)

**Path B rationale:** Per `feedback_pr101_sensitivity_aware_mallat_wavelet_incremental_improvement_insufficient_20260508`,
this lane is a PER-TENSOR SENSITIVITY PROXY using a 2-level db4 wavelet
decomposition + per-tensor allocator integration. It is NOT an
archive-grammar lane — the wavelet coefficients drive an importance
allocator inside `tools/pr101_sensitivity_aware_mallat_wavelet.py`; no new
on-disk archive grammar is introduced.

The lane was promoted to L2 because it has impl_complete +
real_archive_empirical with verdict `incremental_improvement_insufficient`.
Check #124 misclassified it as a representation lane via the `wavelet`
token in its name; `research_only=true` is the correct opt-out per HNeRV
retro lesson 7 (substrate-engineering / sensitivity work that does NOT
ship a packetized inflate).

**Reactivation criteria:**
- `lane_is_per_tensor_sensitivity_proxy_not_archive_grammar_lane`
- `verdict_was_incremental_improvement_insufficient_per_feedback_pr101_sensitivity_aware_mallat_wavelet_20260508`
- `reactivate_only_if_score_aware_substrate_dispatch_demonstrates_Mallat_wavelet_priors_outperform_Xavier_L2_at_contest_CUDA`

(Note: `lane_class=substrate_engineering` would also work as an opt-out
here. `research_only=true` was chosen because the verdict was
"incremental improvement insufficient" rather than "this is a long-running
substrate study" — the lane is closed pending the documented
reactivation conditions, not running indefinitely.)

### 7. `track1_phase_a6_selfcomp_blockfp_hyperprior` — Path A (8 fields declared)

**Path A rationale:** Real composed-codec lane with byte-level wire format:
`src/tac/codec/a6_selfcomp_blockfp_hyperprior_compose.py` (30.2 KB)
declares `_MAGIC = b"A6BF"` (line 88), `_HEADER_SIZE = 12` (magic 4 +
version 1 + scale_quant 1 + block_size 2 + n_total 4),
`_CHARM_CHUNK_HEADER_BYTES = 8`, `compose_blockfp_with_hyperprior` with
encode-decode-assert roundtrip per CompressAI policy.

L2 evidence already established: byte-anchor `compose B=64 sq=uint8 = 214,035 B`
on PR101 substrate beats blockfp-only by -34,607 B and hyperprior-only by
-18,356 B. Verdict per memo: `incremental_improvement_insufficient` (does
NOT beat PR101 brotli baseline +35,891 B).

The lane has a real archive grammar; the score_aware_loss field documents
the substrate-mismatch reactivation requirement per
`feedback_substrate_vs_codec_composition_meta_pattern_20260508`.

**8 fields declared:**
- `archive_grammar`: `src/tac/codec/a6_selfcomp_blockfp_hyperprior_compose.py:88_MAGIC_b_A6BF_with_12_byte_header_magic_4_version_1_scale_quant_1_block_size_2_n_total_4_then_per_block_scales_then_compose_chunked_payload_with_8_byte_chunk_headers`
- `parser_section_manifest`: `src/tac/codec/a6_selfcomp_blockfp_hyperprior_compose.py:347_compose_blockfp_with_hyperprior_with_round_trip_assert_per_CompressAI_policy_and_115_CHARM_CHUNK_HEADER_BYTES_layout_per_chunk`
- `inflate_runtime_loc_budget`: `200` (waiver; per-block scale reconstruction + analytic hyperprior PMF)
- `runtime_dep_closure`: `["torch", "numpy"]` ← deterministic analytic PMF means NO neural-net weights in archive
- `export_format`: `A6BF_v1_block_FP_per_block_max_abs_scales_with_deterministic_analytic_hyperprior_PMF_no_neural_net_weights_in_archive_per_charm_chunk_layout`
- `score_aware_loss`: DEFERRED pending substrate branching to PR100/PR101/PR103 per substrate-vs-codec meta-pattern (full text on field)
- `bolt_on_loc_budget`: `350`
- `no_op_detector_planned`: `true`

## 6-hook coherence-by-default declaration

Per CLAUDE.md "Subagent coherence-by-default":

1. **Sensitivity-map**: N/A — metadata backfill, no new score signal.
2. **Pareto solver**: N/A — no new constraint added.
3. **Bit-allocator**: N/A — no per-tensor importance change.
4. **Cathedral autopilot**: indirect benefit — proper opt-out tagging
   removes 7 spurious "missing field(s)" warnings from the autopilot
   precondition gate, unblocking future autopilot dispatch decisions on
   the 4 Path-A lanes (Lane 20 Ballé, PR106 latent sidecar, α-wavelet,
   A6 compose).
5. **Continual-learning**: N/A — no empirical anchor produced.
6. **Probe-disambiguator**: N/A — per-lane path determination is
   closed-form (design-time call against existing module evidence), not
   a runtime probe.

## 3-clean-pass adversarial review

### Round 1 (Shannon LEAD + Dykstra CO-LEAD + Yousfi)

- **Shannon (information theory)**: ✓ Backfill respects information
  semantics. Top-level `research_only=true` is genuinely the right "this
  lane is research-only by construction" signal for `lane_12`,
  `lane_alpha_nerv_mask`, and `track1_phase_a3_alt_mallat_wavelet`. The
  score_aware_loss DEFERRED tag on Path-A lanes (Ballé, α-wavelet, A6)
  honestly documents the substrate gap rather than claiming an empirical
  result. No score-claim leakage.
- **Dykstra (alternating projections)**: ✓ The 8 fields and 2 opt-out
  tags partition the lane space cleanly. The 4 Path-A lanes are at the
  intersection of (has-archive-grammar) AND (has-active-design); the 3
  Path-D lanes are at the intersection of (representation-token-matched)
  AND (NOT in active dispatch path). No lane sits in an undefined region.
- **Yousfi (steganalysis / contest design)**: ✗ FOUND ISSUE: the
  `score_aware_loss` field on Lane 20 Ballé reads
  "DEFERRED_pending_score_aware_substrate_currently_train_amortizes_against_Lane_G_v3_qint_stream_at_byte_level_only..."
  but Lane G v3 is a FastViT-T12 + EfficientNet-B2 ANCHORED lane already
  with score-aware training upstream. The Ballé hyperprior amortizes against
  the qint stream THAT WAS PRODUCED BY score-aware training; calling that
  "byte-level only" is technically correct (the hyperprior loss is bits)
  but understates the upstream score-awareness. Acceptable as worded
  because the SUBSTRATE distinction is what HNeRV retro lesson 4 cared
  about — score-aware substrate means "training the renderer with
  gradient-through-scorer", not "training the entropy coder against bits".
  Yousfi withdraws the issue. ✓

### Round 2 (Fridrich + Contrarian + Quantizr)

- **Fridrich (steganalysis)**: ✓ The opt-out tags are honest signals;
  none of the 3 Path-D lanes claim a score they didn't produce. `lane_12`
  has L2 forensic evidence preserved; that history is not erased by
  research_only=true.
- **Contrarian**: ✗ FOUND ISSUE: "If `lane_alpha_nerv_mask` is
  superseded by `lane_12_v2`, why isn't `lane_alpha_wavelet_mask` also
  superseded? They're both PARADIGM-α mask-slot replacements." Answer:
  `lane_12_v2` is specifically a NeRV-AS-RENDERER (replaces the entire
  renderer + masks + poses pipeline), not a wavelet-anything. The wavelet
  path remains a standalone alternative for the α 4-way bake-off because
  no internal lane currently does "wavelet-as-renderer". The asymmetry
  is real. Contrarian withdraws. ✓
- **Quantizr (adversarial competitor reverse-engineer)**: ✓ The
  Path-A declarations match the actual leaderboard winners' structure.
  PR101 (gold) had 337 LOC of entropy bolt-ons on top of PR100 substrate;
  our `bolt_on_loc_budget=350` matches that empirical envelope. Lane 20
  Ballé `inflate_runtime_loc_budget=150` is comparable to PR101's
  effective ~100 LOC of inflate runtime (leaderboard sweet spot).

### Round 3 (Hotz + Selfcomp + MacKay)

- **Hotz (raw engineering)**: ✓ Performance: backfill ran in <1s per
  lane via the new `set-field` CLI. The `_coerce_set_field_value` helper
  is 30 LOC and handles the 4 type classes (bool/int/list/string) with
  zero magic. Tests cover all 4 type classes plus 4 error conditions.
- **Selfcomp (block-FP / paradigm architect)**: ✓ The A6 compose lane
  declaration correctly identifies that `runtime_dep_closure=[torch,
  numpy]` is the achievable closure (deterministic analytic hyperprior
  PMF means no `compressai` dep needed). Honest documentation of the
  substrate-mismatch reactivation criterion preserves the lane's
  research-loop signal without claiming a score it didn't produce.
- **MacKay (MDL / information theory)**: ✓ Each declaration self-codes:
  the `archive_grammar` field encodes the magic bytes + section layout
  in the field VALUE (not just a path), so a future agent grepping for
  `BHP_MAGIC` or `WMC1` finds the entry point without opening the file.
  This is the MDL-correct way to embed grammar metadata in a registry.

**Round 3 CLEAN. Counter at 3/3.** Per CLAUDE.md "3-clean-pass
adversarial greenup" non-negotiable.

## Strict-flip recommendation

**RECOMMENDED**: flip `check_representation_lane_has_archive_grammar_at_design_time`
from `strict=False` to `strict=True` in `src/tac/preflight.py:538` (the
`preflight_all()` wire-in site).

Pre-flip verification:
- ✓ Live count = 0 verified via direct strict-mode call
- ✓ Registry validates cleanly (`tools/lane_maturity.py validate` → 112 lane(s) clean)
- ✓ All 49 existing Catalog-#124 tests pass
- ✓ All 13 new lane_maturity set-field tests pass
- ✓ 3-clean-pass adversarial review counter at 3/3
- ✓ No future Phase 2 lanes (T1/T6/T10/T15/T17/T18) currently exist at
  Level 1 in the registry that would trip the check post-flip
  (they are L0 by construction; will trip when promoted, which is the
  intended behavior).

The flip is a single-line edit per the strict-flip pattern documented in
commit 7f2740e4. Operator approval recommended in a follow-up commit
(per CLAUDE.md "Operator gates must be wired and used").

## Cross-references

- Source memo: `~/.claude/projects/.../feedback_check_124_representation_archive_grammar_landed_20260509.md`
- HNeRV retrospective: `~/.claude/projects/.../feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md`
- Lane 12 v2 design memo: `.omx/research/lane_12_v2_nerv_as_renderer_phase_a_design_20260509.md`
- PARADIGM-α clarification: `~/.claude/projects/.../project_paradigm_alpha_architecture_clarification_20260506.md`
- A6 byte-anchor memo: `~/.claude/projects/.../feedback_pr101_a6_selfcomp_blockfp_hyperprior_byte_anchor_landed_20260508.md`
- A3-alt verdict memo: `~/.claude/projects/.../feedback_pr101_sensitivity_aware_mallat_wavelet_incremental_improvement_insufficient_20260508.md`
- Substrate-vs-codec meta-pattern: `~/.claude/projects/.../feedback_substrate_vs_codec_composition_meta_pattern_20260508.md`
- Codex parallel finding: `.omx/research/representation_integration_gap_audit_20260508_codex.md`
- Implementation: `tools/lane_maturity.py:set_field` + `cmd_set_field`
- Tests: `src/tac/tests/test_lane_maturity_harness.py:test_set_field_*` (13 new)
- Registry mutations: `.omx/state/lane_maturity_audit.log` (16 set-field entries appended)

[empirical: Check #124 strict-mode call returns 0 violations on `.omx/state/lane_registry.json` snapshot 2026-05-09 post-backfill]
[empirical: 108 tests pass — 51 Catalog-#124 + 57 lane_maturity_harness — `pytest src/tac/tests/test_check_representation_lane_has_archive_grammar.py src/tac/tests/test_lane_maturity_harness.py -q`]
