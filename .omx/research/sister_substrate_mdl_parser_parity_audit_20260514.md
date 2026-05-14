# Sister-substrate MDL parser parity audit

**Lane**: `lane_ibps1_canonical_surface_promotion_20260514`
**Date**: 2026-05-14 UTC
**Tier**: research_only (audit) — no GPU spend, no score claim

## Question

After IBPS1 (C6 MDL-IBPS) was promoted to a canonical parser-dispatch in
`tac.analysis.hnerv_packet_sections._parse_ibps1_sections` (consuming the
canonical `tac.substrates.c6_e4_mdl_ibps.archive.parse_ibps1_archive_bytes`
+ `IBPS1_SECTION_ROLES` surface), the next question is: **which sister
substrates have an archive magic but are NOT yet recognized by
`ScorerConditionalMDLEstimator`?** Those fall through to a parse error in
`_infer_parser` ("could not infer HNeRV packet parser"), so MDL ablation
on those archives is currently unavailable.

## Findings

`tac.analysis.hnerv_packet_sections` currently recognizes 7 grammars:
PR101 / PR103 / PR106 / A2K1 / A5FC / CPLX1 / IBPS1.

The repository ships 32 substrate archive magics. Of those, the **25 not
yet wired** into the parser-section manifest dispatch are:

| Substrate | Magic | Has parse_archive | Class | Notes |
|---|---|---|---|---|
| `a1_plus_lapose` | `LPA1` | partial | HNeRV-family + sidecar | Wrapper around A1; could route via A1 sub-parser |
| `a1_plus_wavelet_residual` | `WAV1` | partial | HNeRV-family + sidecar | Wrapper around A1; could route via A1 sub-parser |
| `balle_renderer` | `BRV1` | yes | Ballé hyperprior | Class-shift candidate; would benefit from MDL |
| `block_nerv` | `BNV1` | yes | NeRV-family | HNeRV-family; should follow IBPS1 pattern |
| `c1_world_model_foveation` | `WMF\x01` | yes | World-model class-shift | High-priority MDL coverage target |
| `cool_chic` | `CCV1` | yes | Cool-Chic | Across-class learned codec |
| `coord_mlp_residual_sidecar` | `CMLR` | sidecar | HNeRV-family + sidecar | Sidecar wrapper |
| `d1_segnet_margin_polytope` | `D1PY` | yes | Class-shift (SegNet polytope) | High-priority MDL coverage target |
| `d4_wyner_ziv_frame_0` | `WZF\x01` | yes | Class-shift (frame-0 derivation) | High-priority MDL coverage target |
| `driving_prior_world_model` | `DPW1` | yes | World-model class-shift | High-priority MDL coverage target |
| `ds_nerv` | `DSV1` | yes | NeRV-family | HNeRV-family; should follow IBPS1 pattern |
| `ff_nerv` | `FFV1` | yes | NeRV-family | HNeRV-family; should follow IBPS1 pattern |
| `grayscale_lut` | `GLV1` | yes | LUT (Selfcomp lineage) | Across-class |
| `hi_nerv` | `HIV1` | yes | NeRV-family | HNeRV-family; should follow IBPS1 pattern |
| `hybrid_renderer_residual` | `HRR1` | yes | Hybrid renderer | HNeRV+residual; sidecar pattern |
| `pretrained_driving_prior` | `DP1\x00` | yes (canonical compose API) | Pretraining lane | Sister of #210/#211 catalog gates |
| `s2sbs_byte_stuffing` | `S2SB` | yes | Byte-stuffing transform | Codec-pass-through |
| `sabor_boundary_only_renderer` | `SBO1` | yes | Boundary-only renderer | Class-shift candidate |
| `sane_hnerv` | `SHV1` | yes | NeRV-family (PR101-derived) | HNeRV-family; should follow IBPS1 pattern |
| `sar_coherent_pose_pairs` | `SARC` | yes | Pose-coherence renderer | Class-shift candidate |
| `self_compress_nn` | `SCV1` | yes | Self-compression NN | Across-class learned codec |
| `siren` | `SRV1` | yes | SIREN coordinate-MLP | Across-class |
| `tc_nerv` | `TCV1` | yes | NeRV-family | HNeRV-family; should follow IBPS1 pattern |
| `time_traveler_l5_autonomy` | `TT5L` | yes | World-model class-shift | High-priority MDL coverage target |
| `vq_vae` | `VQV1` | yes | VQ-VAE class-shift | Across-class |
| `wavelet` | `WLV1` | yes | Wavelet | Sister of `a1_plus_wavelet_residual` |
| `wyner_ziv_cooperative_receiver` | `WZ1\x00` | yes | Class-shift (cooperative receiver) | Sister of `d4_wyner_ziv_frame_0` |
| `yucr` | `YUCR` | yes | YUV-coherent renderer | Across-class |
| `z3_balle_hyperprior_bolton` | `Z3H1` | no top-level parse | Sidecar bolt-on | Sidecar around base |
| `z4_cooperative_receiver_loss` | `Z4CR` | yes | Class-shift loss | Sister of `wyner_ziv_cooperative_receiver` |
| `z5_predictive_coding_world_model` | `Z5WM` | yes | World-model class-shift | High-priority MDL coverage target |
| `pr95_lora_dora` | `LORA_TRAILER` (u32 trailer, NOT magic prefix) | no top-level parse | LoRA/DoRA bolt-on | Trailer-based; different dispatch shape |

**25 substrates** currently fall through to `_infer_parser`'s final
`HnervPacketSectionManifestError("could not infer HNeRV packet parser")`
when their archive is passed to `ScorerConditionalMDLEstimator.compute()`.
This means **only 7 substrates** (A1/PR101, PR103, PR106, A2K1, A5FC,
CPLX1, IBPS1) get section-aware MDL density on their archives today.

## Why this matters

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable + Catalog
#219 (`check_archive_promotion_blocked_by_mdl_density_above_threshold`),
the autopilot ranker uses scorer-conditional MDL density as a
within-class-vs-across-class signal. A substrate that falls through to
"could not infer" is invisible to that signal.

Per the Z1 ablation finding
(`feedback_z1_mdl_ablation_landed_20260514.md`):

> Tier-A byte-Shannon-entropy density cannot discriminate substrate
> classes when both candidates use brotli on fp16 weights. The
> dispositive test is the **scorer-conditional Tier B/C** MDL, not
> Tier A.

But Tier B/C also needs a working parser-section dispatch to know which
section to perturb. So extending the canonical surface to a sister
substrate is **the prerequisite** for either Tier-A or Tier-B/C ablation
on that substrate's archives.

## Priority ranking for follow-up parser-section dispatch landing

This audit is research-only (no parsers landed here; scope is documented
in the lane charter as "do NOT implement them here; scope-creep"). The
ranking is intended as input to operator-routable decision #1 (below).

| Priority | Substrate | Rationale |
|---|---|---|
| **P0** | `d1_segnet_margin_polytope` | Class-shift; landed Phase 1 trainer + archive |
| **P0** | `d4_wyner_ziv_frame_0` | Class-shift; landed Phase 1; sister of cooperative-receiver |
| **P0** | `pretrained_driving_prior` (DP1) | Pretraining lane used across A1/PR101/HDM8/YUCR/TT5L; high reuse |
| **P1** | `c1_world_model_foveation` | Class-shift; long-term roadmap C1 |
| **P1** | `time_traveler_l5_autonomy` | Class-shift; staircase Step 3+ |
| **P1** | `wyner_ziv_cooperative_receiver` | Class-shift sister of D4 |
| **P1** | `z5_predictive_coding_world_model` | Class-shift; staircase Step 3 |
| **P1** | `driving_prior_world_model` | Class-shift; world-model class |
| **P2** | `balle_renderer` | Substrate-engineering; would inform Z3 hyperprior bolton |
| **P2** | `vq_vae` | Across-class; bit-allocator probe candidate |
| **P2** | `cool_chic` | Across-class learned codec |
| **P2** | `self_compress_nn` | Across-class learned codec |
| **P2** | `siren` | Across-class coord-MLP |
| **P2** | `yucr` | Across-class YUV-coherent |
| **P3** | NeRV-family (tc/block/ff/hi/ds/sane) | Within-class; expected MDL density similar to IBPS1 |
| **P3** | Sidecar wrappers (LPA1/WAV1/CMLR/HRR1/Z3H1) | Routes through underlying parser |
| **P3** | s2sbs_byte_stuffing, sabor_boundary, sar_coherent | Lower-priority class candidates |

## How the pattern works (template for future landings)

Step-by-step pattern from IBPS1 (Part A+B of this same lane):

1. **In `<substrate>/archive.py`**: add a `parse_<MAGIC>_archive_bytes(blob)
   -> dict[name, (start, length)]` function and an
   `<MAGIC>_SECTION_ROLES: dict[str, str]` mapping; export both via the
   substrate's `__init__.py` `__all__`.

2. **In `tac.analysis.hnerv_packet_sections`**:
   - Add `PARSER_<MAGIC> = "<magic>_<family>"` constant
   - Add to `PARSER_CHOICES` tuple
   - Add `<MAGIC>_MAGIC_PREFIX = b"<MAGIC>"` constant
   - Add `if payload.startswith(<MAGIC>_MAGIC_PREFIX): return PARSER_<MAGIC>`
     branch in `_infer_parser` (BEFORE the legacy length-based fallback)
   - Add `if "<magic>" in text or "<substrate>" in text: return
     PARSER_<MAGIC>` text-hint branch
   - Add `_parse_<magic>_sections` helper that imports from the canonical
     surface (LAZY import to avoid forcing substrate dependency on
     `tac.analysis`)
   - Add `if parser_name == PARSER_<MAGIC>: return _parse_<magic>_sections(payload)`
     in `_parse_sections`
   - Add `if parser_name == PARSER_<MAGIC>: return "<human-readable conf>"`
     in `_parser_confidence`

3. **In `tac.analysis.scorer_conditional_mdl`** — NO changes required; it
   delegates to `build_packet_section_manifest` which auto-dispatches.

4. **Add 8+ dedicated tests** mirroring
   `src/tac/substrates/c6_e4_mdl_ibps/tests/test_parse_ibps1_archive_bytes_canonical.py`
   + `src/tac/tests/test_scorer_conditional_mdl_ibps1.py`.

## Outstanding operator-routable decisions (5)

1. **(P0 follow-up)** Promote D1 / D4 / DP1 parsers to canonical surface
   + wire them into `hnerv_packet_sections.py`. These are the 3 highest-
   value class-shift candidates already at L1+ in the lane registry.
   Estimated effort: ~1-2 hours each substrate; $0 GPU.

2. **(P1 batch follow-up)** Class-shift cluster wave: c1_world_model_foveation
   / time_traveler_l5_autonomy / wyner_ziv_cooperative_receiver /
   z5_predictive_coding_world_model / driving_prior_world_model. These
   are all class-shift substrates that benefit MOST from Tier B/C MDL
   ablation (since Tier A byte-entropy alone cannot discriminate them
   per Z1 finding). Estimated effort: ~1 day; $0 GPU.

3. **(P2 batch follow-up)** Across-class learned-codec cluster:
   balle_renderer / vq_vae / cool_chic / self_compress_nn / siren / yucr.
   These would extend the autopilot's class-shift detection to the
   "different encoding family" axis. Estimated effort: ~1 day; $0 GPU.

4. **(P3 batch follow-up)** NeRV-family cluster:
   tc_nerv / block_nerv / ff_nerv / hi_nerv / ds_nerv / sane_hnerv.
   Expected MDL density: ~99% (HNeRV-family ceiling). Useful primarily
   to confirm the within-class hypothesis empirically rather than as
   ranking signal. Lower priority. Estimated effort: ~2-4 hours; $0 GPU.

5. **(Architectural question)** Should the parser-section dispatch in
   `hnerv_packet_sections.py` be refactored from a hand-rolled `if/elif`
   chain into a registry pattern? With 7 grammars today and 25+ pending,
   the dispatch will become unwieldy. Council-grade design decision per
   CLAUDE.md "Design decisions — non-negotiable".

## 6-hook wire-in declarations (Catalog #125 mandatory)

1. **Sensitivity-map contribution** — N/A: this is a parser-discovery
   audit, not a new measurement primitive. The Part B IBPS1 wire-in does
   feed `tac.sensitivity_map.scorer_conditional_entropy_map_v1` via the
   `ScorerConditionalMDLEstimator` it now supports for IBPS1.
2. **Pareto constraint** — N/A: no new R/D constraint introduced.
3. **Bit-allocator hook** — N/A: no per-tensor importance change. Sister
   landings (P0-P3 future waves) WOULD register a bit-allocator hook per
   substrate's section-role taxonomy.
4. **Cathedral autopilot dispatch hook** — IMPLICIT: the autopilot
   ranker already consumes `scorer_conditional_mdl_estimator` results
   for ranking; this audit identifies the next 25 substrates that would
   feed that ranker once their parsers land.
5. **Continual-learning posterior update** — N/A: no empirical anchor in
   this audit.
6. **Probe-disambiguator** — N/A: the audit IS a probe-disambiguator
   data point (which substrates are MDL-visible vs invisible to the
   ranker).

## CLAUDE.md compliance

- **Apples-to-apples evidence discipline**: this is a research-only
  audit; no score claim, no axis tag required.
- **Forbidden /tmp paths**: all paths are repo-relative.
- **No KILL/FALSIFIED verdicts**: the 25 substrates that fall through
  are DEFERRED-pending-parser-promotion, NOT killed.
- **Subagent coherence-by-default**: this audit's findings are wired
  into operator-routable decision #1+#2+#3+#4 (above) so the next
  subagent dispatch has an immediate next-step.
- **Lane maturity registry**: `lane_ibps1_canonical_surface_promotion_20260514`
  is registered at L1 (impl_complete + memory_entry).

## Cross-references

- Source: `feedback_c6_next_wave_landed_20260514.md` (op decisions #3+#4)
- Sister landings: `feedback_mdl_density_gate_and_autopilot_ranker_landed_20260514.md` (Catalog #219)
- Z1 finding: `feedback_z1_mdl_ablation_landed_20260514.md`
- Roadmap: `.omx/research/long_term_multi_year_campaign_roadmap_20260514.md`
- Canonical surface: `src/tac/substrates/c6_e4_mdl_ibps/archive.py::parse_ibps1_archive_bytes`
- Parser dispatch: `src/tac/analysis/hnerv_packet_sections.py::_parse_ibps1_sections`
- XRay primitive: `src/tac/xray/mdl_scorer_conditional.py::ScorerConditionalMDLEstimator`
