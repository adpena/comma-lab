# Codex findings: no-Fourier-basis repository audit

**UTC:** 2026-07-15
**Verdict:** `STRUCTURAL_TRANSITION_BUILT_WARN_ONLY`; strict/default/delete flip `OWED_OPERATOR_GO_N600_BYTE_CLOSED_THROUGHR`.
**Pointer:** `UNMOVED` (`0.19108` submittable / `0.18804` bank).
**UTC:** 2026-07-15  
**Verdict:** `STRUCTURAL_TRANSITION_BUILT_WARN_ONLY`; strict/default/delete flip `OWED_OPERATOR_GO_N600_BYTE_CLOSED_THROUGHR`.  
**Pointer:** `UNMOVED` (`0.19108` submittable / `0.18804` bank).  
**Authority:** `$0` source audit and config/apparatus verification only; no training, dispatch, score, or family-selection claim.

## Exhaustiveness and counts

The sweep covered every text match under `src/tac/` and `experiments/` for case-insensitive `fourier`, explicit `np|numpy|torch|mx|scipy.fft`, and standalone `rfft|irfft|dct|idct` calls. All 114 matching files are Python; there were no matching non-Python files.

- Unique files by primary purpose: **BASIS 86 / TOOL 17 / REPLACEMENT 11 = 114**.
- All matching line-sites, with the two mixed-purpose files split by line: **BASIS 646 / TOOL 144 / REPLACEMENT 132 = 922**.
- `src/tac/information_geometry/`: **0 matches**; untouched.
- `src/tac/canonical_equations/`: inventoried below; untouched because the sibling owner retained that surface.

Classification rule: BASIS means the occurrence selects, generates, encodes, decodes, configures, tests, or preserves a Fourier/DCT/rFFT representation. TOOL means a numerical spectrum/statistic/loss or enforcement apparatus that does not provide decoder features or counted representation coefficients. REPLACEMENT means the curvelet/shearlet implementation, proof, DSL, or family-comparison probe retained to replace Fourier. `texture_trunk.py` and `waterfill_boundary_spectrum_probe.py` are split because their occurrences have different roles.

## BASIS inventory — 646 line-sites

Each row's rationale is the same scoped verdict: representation/config/consumer/replay debt; do not delete or relabel as a curvelet win before the governed A/B.

| File:line(s) | One-line rationale |
|---|---|
| `src/tac/boundary_math/aa_sdf_observation_render.py:33,36,172,193,196,198,205,211,212,216,218,250,324` | Attenuates the active polar directional Fourier feature bank; behavioral receiver math. |
| `src/tac/boundary_math/amortized_luma_carrier.py:19,44,45,60,66,71,77,82,96,97,99,100,112,127,131,155,158,160,267,272,281,298` | Generates and consumes deterministic Fourier coordinate features in a carrier. |
| `src/tac/boundary_math/dseg_aware_fourier_taper.py:2,5,6,9,18,39,82,131,139,141,182,197,198` | Direct Fourier-feature amplitude actuator on the witness basis. |
| `src/tac/boundary_math/keyframe_codec.py:126` | DCT coefficients form the codec representation, not a measurement-only transform. |
| `src/tac/boundary_math/lane_sdf_component.py:370` | Dependency comment identifies the Fourier-basis helper path. |
| `src/tac/boundary_math/lever_b_generator.py:16,42,43,55,56,59,62,63,72,131,138,180,181,211,221,226,231,236,245,258,264,287,290,292,321,326,459,460` | Deterministic Fourier table/features are the decoder representation. |
| `src/tac/boundary_math/lever_b_levelset_generator.py:2,17,21,36,41,43,48,81,84,104,108,124,136,142,176,187,188,191,192,195,232,242,243,245,252,253,255,261,262,266,267,310,311,477,486,487,493,498,508,761,980,1037,1041,1062,1095,1113,1117,1118` | Active polar/directional Fourier front end plus spectral low-pass representation. |
| `src/tac/boundary_math/tests/test_aa_sdf_observation_render.py:24,30,138` | Regression coverage for the Fourier-basis attenuation consumer. |
| `src/tac/boundary_math/tests/test_amortized_luma_carrier.py:27,38,61,62,63,69,81,92,103,106,113,127,128,138,146,153,154,161,162,183,197,226,254,255,258` | Regression coverage for a Fourier-feature carrier. |
| `src/tac/boundary_math/tests/test_dseg_aware_fourier_taper.py:2,19,20,21,91,113,121,129,139,149,157,165,174,183,188,190,192,201,202,203,209,210,215,216,219,220` | Regression coverage for the Fourier taper actuator. |
| `src/tac/boundary_math/tests/test_levelset_yousfi_levers.py:6,21,25,26,34,37,42,45,59,60,61,69,70,74,79,80,81,86` | Regression coverage for polar directional Fourier bank controls. |
| `src/tac/boundary_math/tests/test_lever_b_generator.py:27,28,60,64,65,72,93,96,103,109,115,116,117,124,143,152,238,251` | Regression coverage for Fourier feature generation. |
| `src/tac/boundary_math/texture_trunk.py:343` | Historical comparison explicitly references the base witness Fourier table. |
| `src/tac/boundary_math/tropnnc_witness_reduction.py:171` | Reduction contract consumes fixed polar directional Fourier features. |
| `src/tac/boundary_math/weight_entropy_penalty_mlx.py:13,73` | Rate accounting treats the directional Fourier bank as generated representation state. |
| `src/tac/canonical_equations/__init__.py:147,148,149,450,553` | Registry exports Fourier-taper equations; collision-routed, not edited. |
| `src/tac/canonical_equations/anisotropic_basis_two_regime_allocation_20260707.py:5,102` | Canonical law names a directional Fourier representation. |
| `src/tac/canonical_equations/dseg_aware_fourier_taper_20260709.py:2,4,18,33,49,62,63,73,74,92,96,114,132,140,144,153,160,164,181,184,192,199,202,205,208,219,220` | Canonical law specifies the Fourier taper representation/actuator. |
| `src/tac/canonical_equations/horizon_weighted_margin_20260709.py:213` | Canonical dependency points to the Fourier-taper equation. |
| `src/tac/canonical_equations/lane_dash_residual_root_cause_findings_20260703.py:317` | Canonical finding records an anisotropic Fourier representation. |
| `src/tac/canonical_equations/optimal_basis_selection_20260714.py:24,44,54,64` | Family-selection law retains Fourier candidate/control identities. |
| `src/tac/canonical_equations/witness_measured_findings_20260701.py:304,319` | Historical measured finding names Fourier baseline semantics. |
| `src/tac/canonical_equations/witness_pose_grad_coeff_stability_20260709.py:195` | Canonical dependency points to the Fourier-taper equation. |
| `src/tac/contrib/domain_solvers.py:1753,1768,1770,1798,1837,1839` | Block-DCT coefficients are an encoded solver representation. |
| `src/tac/ffnerv_as_renderer.py:2,6,11,14,17,21,28,30,64,166,217,221,224,250,264,272,309,506` | FFNeRV positional Fourier features are a model representation. |
| `src/tac/flow_compress.py:117` | Inverse DCT participates in compressed flow reconstruction. |
| `src/tac/fp4_quantize.py:424` | Quantization path explicitly includes Fourier-feature representation buffers. |
| `src/tac/local_acceleration/torch_levelset_inflate.py:132` | Inflate parity path reconstructs oriented Fourier features. |
| `src/tac/nerv_mask_codec.py:14` | Codec contract carries a Fourier feature convention. |
| `src/tac/network_codec.py:15,110,112,140,146,184,438,441,454,648,842` | NeRF-style Fourier positional encoding is model input representation. |
| `src/tac/optimization/md_decoupling.py:49,78` | Optimization contract assumes a fixed Fourier table in the representation. |
| `src/tac/optimization/research_basis.py:1804,1805,1820,1823,1828,2056,2063,2079` | Research catalog exposes Fourier features as a basis family; route catalog migration. |
| `src/tac/optimization/scorer_response_prediction.py:94` | Predictor model selection includes Fourier/RBF basis expansions. |
| `src/tac/optimization/substrate_composition_matrix.py:465` | Composition catalog names FFNeRV Fourier-feature representation. |
| `src/tac/residual_basis/coordinate_mlp_residual.py:52` | Residual-basis literature/lineage records Fourier feature representation. |
| `src/tac/scorer_surrogate/replace_round3_fidelity_wall.py:112` | Surrogate preregisters a random Fourier feature matrix. |
| `src/tac/substrates/__init__.py:114` | Substrate registry cites a Fourier-feature model family. |
| `src/tac/substrates/ff_nerv/__init__.py:6` | IDCT reconstructs the substrate's represented RGB coefficients. |
| `src/tac/substrates/ff_nerv/architecture.py:5` | IDCT reconstructs the substrate's represented RGB coefficients. |
| `src/tac/substrates/s2sbs_byte_stuffing/architecture.py:237,258` | rFFT coefficients are encoded/reconstructed residual representation state. |
| `src/tac/substrates/sar_coherent_pose_pairs/__init__.py:23` | Sparse rFFT is the declared pose-pair representation. |
| `src/tac/substrates/sar_coherent_pose_pairs/architecture.py:218,240` | rFFT/irFFT form the pose-pair encoded representation. |
| `src/tac/substrates/sar_coherent_pose_pairs/inflate.py:112` | irFFT decodes counted sparse pose coefficients. |
| `src/tac/substrates/sar_coherent_pose_pairs/score_aware_loss.py:194` | Loss path acts on the substrate's rFFT representation. |
| `src/tac/substrates/siren/activation_family.py:53,189,230,231,401,402,462,490` | Learnable Fourier-series activation is a model basis. |
| `src/tac/substrates/siren/tests/test_nextgen_activations.py:25,205,212,215,232,247,248,262,269,281` | Regression coverage for the learnable Fourier-series activation. |
| `src/tac/tests/test_ffnerv_as_renderer.py:20,58,61,62,68,69,76,77,83,84,86,90,91,150,153` | Regression coverage for FFNeRV Fourier positional encoding. |
| `src/tac/tests/test_island_representation_intrinsic_dim.py:112,114,119,120,124,140,141` | Fourier descriptors are a candidate island representation. |
| `src/tac/tests/test_lever_b_score_native_argmax_smoke.py:65,106,113` | Smoke coverage instantiates the Fourier-feature generator. |
| `src/tac/tests/test_optimal_basis_20260714.py:39,41,52,58,64,66,73,92,116,120` | Basis-family DSL/control regression coverage. |
| `src/tac/tests/test_research_basis.py:88` | Research-basis catalog regression coverage. |
| `src/tac/tests/test_train_witness_realized_through_R_mlx.py:73,151,190,209,211,213,214,255,275,276,284,291,292,303,309` | Witness generator tests instantiate deterministic Fourier features. |
| `src/tac/tests/test_witness_capstone_deepmath_smoke.py:72,76,77,91,92` | Capstone smoke covers directional Fourier feature behavior. |
| `src/tac/tests/test_witness_curriculum_dsl.py:604,772,775` | Curriculum regression covers polar-Fourier scaling passes. |
| `src/tac/torch_vehicle/boundary_routing.py:35,45,46,47,330,332,333` | Optional anisotropic Fourier positional encoding is a model basis. |
| `src/tac/torch_vehicle/driver.py:923` | Driver exposes learnable Fourier-series activation selection. |
| `src/tac/uniward_invariant_enumerator/enumerator.py:852` | JPEG DCT is the represented domain for the enumerator. |
| `src/tac/v2_compose/archive_grammar.py:18` | Archive grammar assumes deterministic Fourier representation state. |
| `src/tac/v2_compose/store_learn_split.py:285` | Store/learn accounting classifies Fourier representation state as generated. |
| `src/tac/witness_dsl/__init__.py:29,78,276,339` | DSL registry exports legacy Fourier controls and campaigns. |
| `src/tac/witness_dsl/basis_control.py:1,3,11,13,21,27,30,34,36` | Explicitly names the retained legacy Fourier A/B control. |
| `src/tac/witness_dsl/campaign.py:39,673,675,689,704,711,724,737,744,763` | Historical campaign compiles polar-Fourier frequency passes. |
| `src/tac/witness_dsl/curriculum_candidate_pool.py:833` | Candidate pool points to the Fourier taper actuator. |
| `src/tac/witness_dsl/curriculum_dsl.py:1937,1938,1942,1945,1947,1948,1950,2306,2310,2318,2319,2335,3068` | DSL retains explicit legacy control plus historical Fourier schedules. |
| `src/tac/witness_dsl/gauge.py:88,224,230,482,486,1567` | Gauge DSL names Fourier bandwidth/IPE representation controls. |
| `src/tac/witness_dsl/optimal_basis_20260714.py:6,35,47,48,49,50,97,100,108,229,240,242,243,296,321,322,323,328,361,363,364,368,371,372,375,385,478,549,566` | Typed family catalog and A/B compiler retain legacy Fourier control semantics. |
| `experiments/build_problem_space_manifest.py:308,327` | Problem-space manifest enumerates Fourier/DCT representation families. |
| `experiments/build_residual_flip_sidecar_pareto.py:109,124,126,141,142,651,655` | Sidecar experiment generates and consumes Fourier features. |
| `experiments/compare_public_pose_manifolds.py:423` | Pose-manifold comparison declares a Fourier/DCT basis type. |
| `experiments/launch_mlx_witness_fleet.py:95` | Fleet launch surface sets Fourier feature count. |
| `experiments/launch_split_by_head_basin.py:255,285` | Launch surface exposes learnable Fourier-series activation. |
| `experiments/plan_yousfi_fridrich_field_equations.py:595` | Planner enumerates Fourier/Walsh subspace representation probes. |
| `experiments/probe_island_representation_intrinsic_dim.py:33,178,204,206,207,215,231,266,711` | Probe evaluates DCT/Fourier descriptors as island representations. |
| `experiments/probe_levelset_gpu_reorient_parity.py:7,8` | Parity probe covers directional Fourier features in the live path. |
| `experiments/probe_mlx_fleet_concurrency_scaling.py:87,170` | Probe launch surface sets Fourier feature count. |
| `experiments/probe_witness_residual_localization.py:55,96,99,114,115,239,283,287` | Residual-localization probe constructs a Fourier-feature generator. |
| `experiments/sg_drf_single_frame_feasibility_probe.py:254,255,430` | Feasibility probe instantiates a Fourier-feature witness. |
| `experiments/test_batched_seed_cograd.py:19` | Historical test notes a different Fourier-basis draw. |
| `experiments/tests/test_levelset_byte_close_and_eval.py:44,106,148,154` | Byte-close regression preserves old Fourier spelling/default bytes. |
| `experiments/tests/test_levelset_checkpoint_resume.py:42,156,160,166,167,575` | Resume regression covers old alias and explicit legacy control custody. |
| `experiments/tests/test_witness_realized_through_R.py:51,65,86,118,181` | Witness tests instantiate Fourier-feature models. |
| `experiments/train_ffnerv_as_renderer.py:44` | Trainer describes Fourier-feature positional encoding. |
| `experiments/train_levelset_witness_realized_through_R_mlx.py:105,106,753,754,823,1156,3753,3754,3995,4025,4053,4057,4080,4081,4082,4092,4096,4127,4157,4161,4162,4241,4265,4323,4498,7422,7430,8901,9585,12460,12461,13124,13125,13127,13128,13152,13153,13158,13162,14539,14542,14545` | Live trainer retains the byte-identical legacy computation only as explicit A/B control. |
| `experiments/train_levelset_witness_realized_through_R_torch.py:1102,1103,1104,1114,1120` | Torch trainer consumes the Fourier taper path. |
| `experiments/train_substrate_ff_nerv.py:11` | Trainer's IDCT reconstructs represented coefficients. |
| `experiments/train_witness_realized_through_R.py:45,93,106,111,112,113,119,132,136,140,144,145,146,156,157,458,459,562,563,607,638,642` | Torch witness uses deterministic Fourier coordinate features. |
| `experiments/train_witness_realized_through_R_mlx.py:38,301,314,316,317,318,368,382,387,391,409,410,433,437,455,456,502,998,1453,1687,1714,1722,1737,1749,1750,1770,2204,2205,2235,2738,2739,2821,2825,2979,3100` | MLX witness uses deterministic Fourier coordinate features. |

## TOOL inventory — 144 line-sites

These transforms measure spectra/statistics, define a loss, or enforce provenance. They do not provide decoder features or counted basis coefficients and must be kept. Same-line waivers were added only on owned, behavior-neutral sites; remaining waiver debt is routed.

| File:line(s) | One-line rationale |
|---|---|
| `src/tac/analysis/scorer_spectral_sensitivity_v2.py:47,213,214,231,232,289,290,528,537` | FFT measures scorer spectral sensitivity/anisotropy. |
| `src/tac/boundary_math/texture_trunk.py:190,191` | FFT measures realized texture spectrum; owned lines now carry substantive waivers. |
| `src/tac/canonical_equations/fresh_frequency_shift_init_20260712.py:200` | FFT measures frequency content for initialization selection. |
| `src/tac/fridrich_losses.py:63` | Documentation distinguishes an explicit matrix from an FFT implementation. |
| `src/tac/losses/core.py:2164` | rFFT computes a spectral orthogonality loss, not decoder features. |
| `src/tac/preflight.py:1442,1445,1448,5455,6471,28774` | Preflight wrapper/apparatus references the structural gate and unrelated tool checks. |
| `src/tac/research/geometry_deliberation.py:679,705,822,823,825,832,837,844,1313` | DFT basis is used to measure Hessian/eigenvector alignment. |
| `src/tac/research/karpathy_cnn_residual_analysis.py:65` | rFFT measures CNN residual spectral content. |
| `src/tac/residual_basis/siren_residual.py:182,183,206` | FFT computes diagnostic radial-frequency buckets for a SIREN residual. |
| `src/tac/sfess_cached_replay.py:327` | FFT evaluates a Poisson-binomial PMF statistic. |
| `src/tac/tests/test_check_pose_basis_fit_kill.py:9,95,97,100,107` | Synthetic enforcement fixtures for the separate pose basis-fit guard. |
| `src/tac/tests/test_fresh_frequency_shift.py:27,39,50,54,64` | FFT constructs synthetic measurement fixtures; owned lines now carry waivers. |
| `src/tac/tests/test_scorer_spectral_sensitivity_v2.py:92,93,102,264` | FFT constructs/measures synthetic spectral-sensitivity fixtures; waivered. |
| `src/tac/tests/test_v9_provenance_gates.py:17,21,68,69,198,283,284,314,318,322,324,327,328,331,333,334,336,339,343,345,351,352,355,358,363,366,369,372,376,394,401,404,406,416,421,422,424,427,429,430,432,435,437,438,441,442,559` | Synthetic positive/negative fixtures and assertions for enforcement apparatus. |
| `src/tac/through_r/stem_perception.py:202` | FFT measures the scorer stem's spectral response. |
| `src/tac/v9_provenance_gates.py:150,151,161,163,164,165,166,167,168,172,217,299,303,304,310,316,342,353,362,366,371,372,376,383,388,390,704,750,755,756,764,775,776,780,813,820,825,829,835,841,844,955,959,963` | Signature table, audit engine, claim validation, and exports are enforcement apparatus. |
| `src/tac/witness_init/fresh_frequency_shift.py:132` | FFT measures frequency content for initialization selection. |
| `experiments/waterfill_boundary_spectrum_probe.py:102,103,146` | FFT measures boundary-patch spectra inside a replacement-family comparison. |

## REPLACEMENT inventory — 132 line-sites

These are curvelet/shearlet implementations, proofs, DSL surfaces, or comparison probes. Keep them; their presence is not a realized-through-R family verdict.

| File:line(s) | One-line rationale |
|---|---|
| `src/tac/boundary_math/compact_shearlet_frame.py:13,16,27,203,241,242,258,264,305,389,391,395,397,398,405,406,443,448,453` | Genuine compact-shearlet replacement and anti-Fourier structural proof. |
| `src/tac/boundary_math/tests/test_compact_shearlet_frame.py:5,31,34,36,37` | Replacement localization/shear-steering regression proof. |
| `src/tac/boundary_math/tests/test_windowed_curvelet_frame.py:5,16,18,19,34,37,39,41,44,45,47,48,61` | Replacement curvelet regression and anti-alias proof. |
| `src/tac/boundary_math/windowed_curvelet_frame.py:5,14,159,197,201,210,214,274,277,279,283,285,286,293,294,309,310,313` | Windowed-curvelet replacement implementation and parity comparison. |
| `src/tac/canonical_equations/deepmath_amortizing_argmax_laws_20260704.py:177,492,500,504` | Curvelet approximation-rate replacement law contrasted with Fourier. |
| `src/tac/canonical_equations/windowed_curvelet_parabolic_capacity_20260714.py:17,25,28,31,32,34,67,68,70,107,110,113,117,118,122,127,133,137,139,204,207,231,233,242,279` | Curvelet-capacity replacement equation; inventory only, untouched. |
| `src/tac/witness_dsl/tests/test_windowed_curvelet_basis_lever.py:5,54,99,104` | Selectable replacement DSL and legacy-control A/B regression. |
| `src/tac/witness_dsl/windowed_curvelet_basis_lever_20260714.py:34,36,83` | Selectable windowed-curvelet replacement lever, default OFF pending measurement. |
| `experiments/curvelet_vs_fourier_capacity_probe.py:2,23,33,46,48,49,158,159,160,163,169,188,193,203,204,206` | $0 replacement-capacity comparison; upper bound only. |
| `experiments/shearlet_vs_curvelet_vs_fourier_capacity_probe.py:1,5,6,14,28,48,50,51,71,96,97,98,110,117,120,144,155,167,177` | $0 replacement-family comparison; upper bound only. |
| `experiments/waterfill_boundary_spectrum_probe.py:2,10,28,88,92,196` | Replacement-family waterfill comparison; FFT measurement lines are classified TOOL above. |

## Structural transition and round-1 adversarial verdict

- The parser and DSL canonicalize the old `polar_fourier` spelling to `legacy_fourier_ab_control`; old checkpoints/CLI remain readable and the historical control omits the additive basis key exactly as before.
- The legacy path emits `LEGACY_AB_CONTROL_ONLY`; it is not presented as a ship default or curvelet label.
- `v9_ideal_mod32_basis_ab_configs()` returns a pure typed control/treatment pair. Excluding `--out-dir`, the compiled maps differ only by `--basis`.
- `collect_live_v9_fake_claims()` derives the basis label from the compiled `--basis`; V9 no longer labels an implicit global Fourier implementation as curvelet.
- The new source gate catches an executable synthesized Fourier feature call even with an inline comment and an unregistered CamelCase Fourier encoder; valid FFT-tool waivers and replacement files pass; absent/placeholder waivers fail; strict mode raises; `preflight_all` calls it with `strict=False`.
- Replacement modules are not whole-file exempt: the gate derives exemptions from the parsed AST only for the named top-level certificate class and proof function on each exact canonical frame path. The actual `windowed_curvelet_feats` and `compact_shearlet_feats` functions remain audited, and same-named scopes at noncanonical paths are caught.
- Gate apparatus fixtures are exact-path exempt from the live scan, not signature-wide exempt. This prevents the positive test case from becoming permanent live debt while preserving the synthesized violation test.
- Round-1 re-derivation found and fixed two risks: inline comments initially hid executable prefixes, and generated receiver-source spelling was initially at risk of changing the byte-identical OFF path. The final code scans executable prefixes and leaves the generated legacy receiver source/bytes unchanged.

## Safe swaps, routes, and blockers

- Behavior-changing safe swaps completed: **0**.
- Behavior-neutral annotations completed: owned `texture_trunk` measurement and transform-only tests carry substantive `FFT_TOOL_USE_OK` waivers. Replacement certificate/control proof scope is encoded only in the gate AST policy; the frame modules carry no new annotations.
- Routed, not touched: canonical equations; all other live/historical boundary decoders; research-basis catalog; legacy trainers/substrates; remaining tool-waiver owners.
- Exact blocker to strict ban/default flip/delete: missing operator-GO n600 byte-closed realized-through-R no-d_seg-regression verdict for `windowed_curvelet` versus `legacy_fourier_ab_control`.
- Exact family verdict: **OPEN**. Existing capacity/receiver evidence is not the required matched byte-closed shipping-vehicle authority.
- Pointer remains unchanged; no action in this audit can promote a score.

## Durable routing

- DAG FEED: `.omx/research/no_fourier_basis_DAG_FEED_20260715.md`.
- Runnable typed A/B receipt: `.omx/research/curvelet_throughR_basis_ab_receipt_20260715.json` (`PREPARED_NOT_FIRED_OPERATOR_GO_REQUIRED`).
