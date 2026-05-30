---
council_tier: T1
council_attendees: [Implementation-Agent]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Rev #3 + #4 substrate-engineering scaffolding is valid even though M12a Z8_TRAINER_MODE=full does NOT route through canonical_quadruple_binding"
    classification: HARD-EARNED
    rationale: "Per recipe DEFERRED note + Yousfi memo Axis 1+2+3 Rev #3+#4+#5 are EXPLICITLY classified M12c+ scope; the operator task description scopes THIS lane to substrate-engineering scaffolding NOT M12a runtime activation; per CLAUDE.md NO FAKE IMPLEMENTATIONS the scaffolding is honest substrate-engineering for future M12c routing"
  - assumption: "deterministic_pose_proxy_6dim_compress_time_fallback is an HONEST proxy not a FAKE PoseNet output"
    classification: HARD-EARNED
    rationale: "Per Catalog #287 NO PLACEHOLDER + CLAUDE.md NO FAKE IMPLEMENTATIONS: the proxy is explicitly labeled in pose_source field; downstream consumers cannot mistake it for PoseNet; the M12c+ paid-CUDA scope replaces it with actual PoseNet via load_default_scorers per canonical pattern"
  - assumption: "Z8_TRAINER_MODE=full at M12a means the canonical_quadruple_binding code path is DEAD CODE not invoked at M12a"
    classification: HARD-EARNED
    rationale: "Empirically verified by reading experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py main() lines 557-563: full mode routes to _full_main which calls run_mlx_score_aware_full_main NOT canonical_quadruple_binding; the canonical_quadruple_binding code path is ONLY invoked when --canonical-quadruple-binding flag is passed (and Z8_TRAINER_MODE=full does NOT pass it)"
  - assumption: "Updating M12a predicted_band from [0.175, 0.190] to [0.150, 0.175] without Rev #3+#4 being ACTIVE in trainer at M12a would be a phantom-score claim"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md NO FAKE IMPLEMENTATIONS + Catalog #324 predicted-band-validation discipline: the deep-Yousfi-grounded [0.150, 0.175] band is achievable IF all 5 revisions wired into ACTIVE trainer; per CLAUDE.md NO PHANTOM SCORES the M12a band MUST track the actual M12a code path (Z8_TRAINER_MODE=full + Rev #1+#2 only); the deep band is M12c-conditional reference documented in predicted_band_m12c_deep_yousfi_grounded_reference field"
council_decisions_recorded:
  - "Rev #3 PoseNet 6-dim Wyner-Ziv side_info per canonical equation #150 LANDED as opt-in feature in canonical_quadruple_binding.py"
  - "Rev #4 4-level Mallat pyramid support verified at config level (existing per-level loop handles num_levels=4 natively; no code change needed beyond test coverage)"
  - "Rev #5 recipe predicted_band M12a unchanged [0.175, 0.190]; deep-Yousfi-grounded [0.150, 0.175] reference documented in predicted_band_m12c_deep_yousfi_grounded_reference + reactivation path"
  - "26 dedicated tests pass + 252 baseline z8 tests pass = 278/278 total"
  - "macOS-CPU advisory smoke landed at experiments/results/z8_yousfi_revisions_3_4_5_smoke_20260530T184139Z/yousfi_revisions_smoke_output.json"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: ""
horizon_class: plateau_adjacent
canonical_equation_references:
  - wyner_ziv_decoder_side_information_rate_savings_v1
mlx_local_advisory_only: true
axis_tag: "[macOS-CPU advisory]"
score_claim: false
promotable: false
evidence_grade: macOS-CPU-advisory
---

# Z8 M12a Yousfi Revisions #3 + #4 + #5 substrate-engineering scaffolding LANDED

**UTC**: 2026-05-30
**Lane**: `lane_z8_yousfi_revisions_3_4_5_20260530`
**Source memo**: `.omx/research/council_yousfi_voice_canonical_inverse_steganalysis_review_z8_m12a_modal_t4_l2_long_training_pre_dispatch_20260530.md` (commit `843b4bfd8`)
**Mission contribution**: `apparatus_maintenance` (Catalog #300)

## Summary

Per the Yousfi voice canonical inverse-steganalysis review memo `843b4bfd8` (2026-05-30) Yousfi voice returned PROCEED_WITH_REVISIONS with 5 canonical revisions. Revisions #1 + #2 (Z8_TRAINER_MODE=full + Z8_M7_SOURCE=empirical_from_master_gradient) landed in commit `0b6a3793d`. THIS lane lands the 3 remaining revisions as substrate-engineering scaffolding for M12c+ scope per recipe DEFERRED notes lines 286-297.

## Rev #1+#2 ARE ACTIVE at M12a (sister landing commit `0b6a3793d`)

Per recipe env_overrides post-`0b6a3793d`:
- `Z8_TRAINER_MODE: "full"` — routes through `_full_main` → `run_mlx_score_aware_full_main` (canonical EMA + canonical eval_roundtrip + canonical Hinton-KL T=2.0)
- `Z8_M7_SOURCE: "empirical_from_master_gradient"` — binds M7 Path B2

## Rev #3+#4+#5 are M12c-SCOPE SCAFFOLDING (this lane)

Per CLAUDE.md NO FAKE IMPLEMENTATIONS + recipe DEFERRED note: Rev #3+#4+#5 are landed as substrate-engineering scaffolding for future M12c routing. They do NOT change M12a runtime behavior because Z8_TRAINER_MODE=full routes through `_full_main` which uses `run_mlx_score_aware_full_main` NOT `canonical_quadruple_binding`. The canonical_quadruple_binding compose pattern is ONLY invoked when `--canonical-quadruple-binding` flag is passed (not Z8_TRAINER_MODE=full path).

### Revision #3: PoseNet 6-dim Wyner-Ziv side_info per canonical equation #150

**File changes**:
- `src/tac/substrates/z8_hierarchical_predictive_coding/canonical_quadruple_binding.py`:
  - Added `POSE_SIDE_INFO_DIM = 6` canonical constant (CLAUDE.md "Exact scorer architectures" PoseNet pose vector dim).
  - Added `_POSE_SIDE_INFO_PROJECTION_SEED_DEFAULT = 17` (encoder+decoder MUST agree).
  - Added `_project_pose_6dim_to_side_info_shape(pose, side_info_shape, projection_seed)` — deterministic Gaussian projection from (B, 6) → (B, side_c, side_h, side_w).
  - Added `_deterministic_pose_proxy_6dim(pair_rgb_target)` — HONESTLY LABELED 6-dim spatial-statistics proxy (mean per channel + spatial-gradient L2 norm) used when actual PoseNet weights are NOT loaded at the M9 macOS-CPU smoke surface. NEVER claimed to be PoseNet; pose_source field labels it `deterministic_pose_proxy_6dim_compress_time_fallback`.
  - Added `compute_pose_side_info_canonical_equation_150(pair_rgb_target, side_info_shape, pose_6dim_batch=None, projection_seed)` — the canonical public API. Returns `(side_info, pose_6dim, pose_source)` tuple per Catalog #287 NO PLACEHOLDER discipline.
  - Wired `pose_side_info_canonical_equation_150_enabled` opt-in kwarg through `canonical_quadruple_forward_step` AND `run_canonical_quadruple_training_loop` (default OFF preserves backward compat).
  - Wire emits `wyner_ziv_side_info_source` + optional `wyner_ziv_pose_6dim` keys in forward result dict.

**Rev #3 honest scope**: per CLAUDE.md "Strict scorer rule" + Catalog #6 PoseNet is FORBIDDEN at inflate time. The canonical equation #150 surface requires PoseNet at COMPRESS time only; the per-pair 6-dim vector is serialized into archive bytes; decoder reads vector from archive (NOT recomputed). For the M12c+ paid-CUDA scope: encoder calls real PoseNet via `load_default_scorers()`; decoder reads vector. For THIS M9 macOS-CPU smoke surface: PoseNet weights are not loaded so the helper falls back to deterministic 6-dim spatial-statistics proxy honestly labeled per Catalog #287.

### Revision #4: 4-level Mallat pyramid below SegNet 256x192 blind-spot

**Verification**: existing per-level loop in `canonical_quadruple_forward_step` handles `num_levels=4` natively — no code change needed. At contest eval_size (384, 512):
- Level 0: 384×512  (above 256×192)
- Level 1: 192×256  (above 256×192 — boundary case)
- Level 2: 96×128   (below 256×192)
- Level 3: 48×64    (clearly below SegNet stride-2 stem blind-spot)

Per CLAUDE.md "Exact scorer architectures": SegNet EfficientNet-B2 stride-2 stem loses half resolution immediately → artifacts below (256, 192) are structurally invisible. The canonical 4-level Mallat pyramid extends the Z8 substrate's per-level coverage into the SegNet blind-spot resolution band where Wyner-Ziv conditional coding can spend bytes SegNet cannot "see".

**Tests confirm**: `Z8HierarchicalConfig(num_levels=4, ...)` builds 4 per-level adapters; forward_step emits 4 per-level losses + 4 wavelet subband norms; training loop produces valid 4-level artifact; at contest resolution Level 3 wavelet subband shape = (48, 64) confirmed below blind-spot.

### Revision #5: recipe predicted_band documents M12a vs M12c bands honestly

**File change**: `.omx/operator_authorize_recipes/substrate_z8_hierarchical_predictive_coding_modal_t4_dispatch.yaml`:
- `predicted_band: [0.175, 0.190]` UNCHANGED (M12a band per Rev #1+#2 partial scenario per Yousfi memo Axis 5)
- ADDED `predicted_band_m12c_deep_yousfi_grounded_reference: [0.150, 0.175]` (deep-Yousfi-grounded scenario per Yousfi memo Axis 5)
- ADDED `predicted_band_m12c_reactivation_path` field documenting the canonical routing wave required to activate Rev #3+#4 at M12c
- ADDED `predicted_band_m12c_deep_yousfi_grounded_reference_contest_cuda: [0.164, 0.189]` (sister CPU-CUDA gap projection)
- Updated `predicted_band_reactivation_criterion` to honestly cite Catalog #287 NO FAKE IMPLEMENTATIONS rationale for keeping M12a band unchanged

**Why M12a band stays at [0.175, 0.190]**: per CLAUDE.md NO FAKE IMPLEMENTATIONS + Catalog #324 predicted-band-validation discipline. Updating the M12a band to the deep-Yousfi-grounded [0.150, 0.175] when Z8_TRAINER_MODE=full does NOT route through Rev #3+#4 active code would be a phantom-score claim. The deep band is documented as M12c-conditional reference.

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| PoseNet 6-dim helper | ADOPT_CANONICAL via CLAUDE.md "Exact scorer architectures" | 6-dim is canonical PoseNet output per upstream scorer; not a fork-able choice |
| Pose-proxy fallback | FORK_BECAUSE_PRINCIPLED_MISMATCH | The macOS-CPU smoke surface cannot load PoseNet weights without contest scorer import; the proxy is per-Z8-substrate per Catalog #290 substrate-optimal engineering |
| Projection seed = 17 | FORK_BECAUSE_PRINCIPLED_MISMATCH | Need a deterministic seed distinct from M6 Wyner-Ziv projection seed (default 0); 17 is arbitrary but pinned per Catalog #287 |
| 4-level Mallat config | ADOPT_CANONICAL via existing per-level loop | The canonical adapter loop handles arbitrary num_levels via Z8HierarchicalConfig field; no fork needed |
| M12a predicted_band | ADOPT_CANONICAL via Rev #1+#2 partial scenario | Per Yousfi memo Axis 5 partial scenario; the deep scenario requires Rev #3+#4 ACTIVE per Catalog #324 |

## 9-dimension success checklist evidence

1. **UNIQUENESS**: per-substrate canonical equation #150 instantiation (PoseNet 6-dim → side_info_shape projection) is Z8-specific scaffolding
2. **BEAUTY + ELEGANCE**: 3 small canonical helpers (`_deterministic_pose_proxy_6dim`, `_project_pose_6dim_to_side_info_shape`, `compute_pose_side_info_canonical_equation_150`); opt-in kwarg preserves backward compat; 30-second reviewable per surface
3. **DISTINCTNESS**: explicitly distinct from prior top-LL spatial-mean wiring; pose_source label disambiguates at every callsite
4. **RIGOR**: 26 dedicated tests; pseudo-inverse round-trip verifies projection preserves rank-6 information; 4-level config verified against contest resolution scorer blind-spot
5. **OPTIMIZATION PER TECHNIQUE**: per Catalog #290 the canonical Wyner-Ziv 1976 Theorem 1 R(X|Y) << R(X) discipline; PoseNet 6-dim is the canonical Y for dashcam ego-motion video per canonical equation #150
6. **STACK-OF-STACKS-COMPOSABILITY**: Rev #3 + Rev #4 independently activatable; Rev #5 documents both individually + combined paths
7. **DETERMINISTIC REPRODUCIBILITY**: encoder+decoder share projection seed 17; pose proxy is byte-deterministic given pair RGB input; 4-level Mallat is exact-round-trip per Mallat 1989 §7.5
8. **EXTREME OPTIMIZATION**: each helper is <30 LOC; opt-in kwarg has zero overhead when disabled
9. **OPTIMAL MINIMAL CONTEST SCORE**: M12a band UNCHANGED per Catalog #287 NO FAKE IMPLEMENTATIONS; M12c deep band [0.150, 0.175] documented as reactivation path

## Observability surface (Catalog #305)

- **Inspectable per layer**: `wyner_ziv_side_info_source` + `wyner_ziv_pose_6dim` keys in forward result dict
- **Decomposable per signal**: Rev #3 affects only M6 Wyner-Ziv path (not M8 per-level loss; not M4 Mamba-2 state); Rev #4 affects per-level decomposition pyramid count
- **Diff-able across runs**: smoke artifact carries per-variant per-epoch loss trajectory + per-variant final wyner_ziv_payload_bytes for explicit diff vs baseline
- **Queryable post-hoc**: smoke output JSON has canonical schema `z8_yousfi_revisions_3_4_5_smoke_output_v1` with per-variant + provenance fields
- **Cite-able**: every variant carries `lane_id` + `subagent_id` + commit citation in canonical_provenance
- **Counterfactual-able**: opt-in kwargs allow byte-mutation per Catalog #139 (toggle Rev #3 on/off; toggle num_levels 3↔4)

## Cargo-cult audit per assumption

1. **The pose-proxy is a legitimate stand-in for PoseNet at smoke time** — HARD-EARNED per Catalog #287 NO PLACEHOLDER: the proxy is honestly labeled in pose_source field; never claims to BE PoseNet; the M12c+ scope replaces with real PoseNet via canonical `load_default_scorers()` pattern
2. **Projection seed 17 is arbitrary but pinned** — HARD-EARNED per Catalog #287: arbitrary deterministic seeds are accepted IF pinned + deterministic (encoder+decoder agree); the seed CANNOT be 0 (collides with M6 default); 17 is a prime number deterministic alternative
3. **4-level Mallat at contest resolution puts Level 3 below SegNet blind-spot** — HARD-EARNED per CLAUDE.md "Exact scorer architectures": SegNet EfficientNet-B2 stride-2 stem loses half resolution; 48×64 < 256×192 is unambiguous structural blind-spot
4. **M12a band stays at [0.175, 0.190] not [0.150, 0.175]** — HARD-EARNED per CLAUDE.md NO FAKE IMPLEMENTATIONS + Catalog #324: Z8_TRAINER_MODE=full at M12a does NOT route through canonical_quadruple_binding; updating M12a band would be phantom-score
5. **Adding pose_side_info breaks the loss decrease pattern** — CARGO-CULTED until empirically tested → empirical SMOKE shows rev3 and baseline produce IDENTICAL loss trajectory because M8 per-level loss is invariant to side_info (only M6 round-trip depends on it); confirmed honest

## 5-axis empirical falsification check per assumption taxonomy

Per Slot EEE audit (`feedback_slot_eee_fake_implementation_audit_on_today_l0_scaffolds_per_operator_binding_must_review_for_fake_implementations_landed_20260529.md`):

- **Cite-vs-impl**: ✓ helper imports + tests verify actual function output, not fixture-inverted
- **Test substance**: ✓ 26 tests cover positive + negative + waiver + invariant cases
- **Smoke realism**: ✓ smoke uses REAL `upstream/videos/0.mkv` per Catalog #213 (not 32×32 random noise per Slot EEE finding)
- **Predicted-band grounding**: ✓ M12a band UNCHANGED per Catalog #287; deep band documented as M12c-reactivation reference not auto-active
- **Strategy-enum non-degeneracy**: ✓ pose_source has 3 distinct values (top_ll spatial mean / caller-supplied / proxy fallback) each with distinct semantic

## Smoke artifact

- Path: `experiments/results/z8_yousfi_revisions_3_4_5_smoke_20260530T184139Z/yousfi_revisions_smoke_output.json`
- Schema: `z8_yousfi_revisions_3_4_5_smoke_output_v1`
- 4 variants exercised: baseline / Rev #3 only / Rev #4 only / Rev #3+#4 combined
- All 4 variants: `CONVERGED_MONOTONIC` with finite Wyner-Ziv payload bytes
- Wall-clock: <1s per variant (M9 OPTIMIZER-FREE smoke at 32×32 × 4 pairs × 5 epochs)
- Axis tag: `[macOS-CPU advisory]` per Catalog #192
- Score claim: `False`; promotable: `False`

## NO FAKE IMPLEMENTATIONS attestation per CLAUDE.md non-negotiable

- ✓ pose proxy honestly labeled `deterministic_pose_proxy_6dim_compress_time_fallback` in pose_source field
- ✓ PoseNet weights NOT loaded at smoke surface — explicitly stated in attestation block
- ✗ gradient descent NOT active — explicitly stated in attestation block; this is the Yousfi memo Axis 4 OPTIMIZER-FREE finding the operator already knows; M9 binding-integration milestone is OPTIMIZER-FREE by design per docstring line 702; canonical optimizer wiring is M12c+ scope per recipe DEFERRED note
- ✓ opt-in default OFF preserves backward compat (252 baseline tests pass)
- ✓ M12a band unchanged per Catalog #287; deep band documented as M12c-conditional

## Per-revision verdicts

| Revision | Status | Rationale |
|---|---|---|
| Rev #3 PoseNet 6-dim Wyner-Ziv side_info | **LANDED-AS-OPTIN-SCAFFOLDING** | Canonical equation #150 instantiation as opt-in kwarg; not active at M12a; future M12c routing activates |
| Rev #4 4-level Mallat below SegNet blind-spot | **LANDED-AS-CONFIG-SUPPORT** | Existing per-level loop natively handles num_levels=4; verified via 5 dedicated tests; not active at M12a |
| Rev #5 recipe predicted_band update | **LANDED-AS-DEEP-REFERENCE-DOCUMENTATION** | M12a band UNCHANGED per Catalog #287 NO FAKE IMPLEMENTATIONS; deep band added as M12c-conditional reference; reactivation path documented |

## Operator-routable next steps

1. **M12a paid Modal dispatch with Rev #1+#2 active** (this is unchanged; Rev #3+4+5 scaffolding does not affect M12a):
   ```bash
   .venv/bin/python tools/operator_authorize.py \
       --recipe substrate_z8_hierarchical_predictive_coding_modal_t4_dispatch \
       --platform modal \
       --gpu T4
   ```
2. **M12c routing wave** (future; activates Rev #3+#4 in trainer's active code path): operator-routable sister substrate-engineering wave to either (a) make Z8_TRAINER_MODE=canonical_quadruple_v2 the canonical mode at M12c, OR (b) wire canonical_quadruple_binding compose pattern into `_full_main` as the inner forward pass. Estimated wall-clock: 1-3h substrate-engineering per HNeRV parity L7.
3. **Optimizer wiring at M12c**: per Yousfi memo Axis 4 the canonical_quadruple_binding M9 path is OPTIMIZER-FREE; activating Rev #3+#4 productively requires wiring an optimizer + EMA + eval_roundtrip per CLAUDE.md non-negotiables. This is the OPERATOR-ROUTABLE for M12c+ scope item the recipe explicitly defers.

## 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map = ACTIVE (Rev #3 wires canonical equation #150 sensitivity surface via PoseNet 6-dim side_info)
- hook #2 Pareto constraint = N/A (defensive validator gate; no Pareto-relevant signal contribution beyond existing M6 R(X|Y) bound)
- hook #3 bit-allocator = ACTIVE (Rev #3 side_info path may affect bit allocation across Wyner-Ziv archive sections at M12c+)
- hook #4 cathedral autopilot dispatch = N/A (scaffolding lands; no candidate routing affected at M12a)
- hook #5 continual-learning posterior = ACTIVE (canonical equation #150 + canonical Provenance per Catalog #323; smoke artifact carries posterior anchor)
- hook #6 probe-disambiguator = ACTIVE (the `wyner_ziv_side_info_source` label IS the canonical disambiguator between prior-top-LL / Rev-#3-pose-proxy / Rev-#3-real-PoseNet wiring)

## Catalog #348 retroactive sweep companion memo

`.omx/research/retroactive_sweep_for_yousfi_revisions_3_4_5_substrate_engineering_20260530.md`

## Sister cross-references

- Yousfi voice review memo: `.omx/research/council_yousfi_voice_canonical_inverse_steganalysis_review_z8_m12a_modal_t4_l2_long_training_pre_dispatch_20260530.md` (commit `843b4bfd8`)
- Rev #1+#2 sister landing: commit `0b6a3793d` (recipe env_overrides + CLAUDE.md NO FAKE IMPLEMENTATIONS non-negotiable)
- Catalog #325 symposium memo: `.omx/research/council_t3_grand_council_per_substrate_symposium_z8_hierarchical_predictive_coding_m12_paid_modal_t4_l2_long_training_plus_paired_cuda_canonical_sub_0_189_attempt_20260530.md` (commit `4bcc84fc0`)
- Canonical equation #150: `wyner_ziv_decoder_side_information_rate_savings_v1` (registered Wave N+36 commit `c2780c7ba`)
- M9 binding-integration anchor: commit `bb48f691c`
- M10 inflate-consumes-real-trained-weights anchor: commit `59bdf9c93`
- M11 cycle-closure anchor: commit `2f8570755`

<!-- Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com> -->
