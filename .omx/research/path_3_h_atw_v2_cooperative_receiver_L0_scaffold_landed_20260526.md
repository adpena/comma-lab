<!-- SPDX-License-Identifier: MIT -->
---
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Atick
  - Redlich
  - Ballard
  - Tishby
  - Wyner
  - AssumptionAdversary
  - Contrarian
  - PR95Author
  - Yousfi
  - Fridrich
council_quorum_met: true
council_verdict: PROCEED
council_assumption_adversary_verdict:
  - assumption: "Path (b) FORK with Layer 1 META-unwind landed structurally per Phase 3 design + L0 SCAFFOLD"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "47/47 tests pass; substrate identity constants exposed inline per Catalog #241 + #124; archive grammar magic ATWv2CR2 distinct from sister substrates; cdf_table_blob REMOVED from canonical grammar (Phase 1 CC-8 unwind operationalized); ego-motion FOE projection conditioning surface IMPLEMENTED + byte-mutation tests verify operational consumption per Catalog #139/#272/#220; MLX↔numpy parity test passes (when MLX available); numpy reference fully portable per Axis 3; _full_main raises NotImplementedError per Catalog #240(c) pending Phase 4 council approval."
council_decisions_recorded:
  - "op-routable #1: Phase 4 D4-equivalent probe for ego-motion conditioning surface (Phase 4 deliverable)"
  - "op-routable #2: Phase 4 PyTorch training loss via canonical Atick-Redlich primitive (Phase 4 deliverable)"
  - "op-routable #3: Phase 4 MLX→PyTorch state_dict export bridge per #1251 pattern (Phase 4 deliverable)"
  - "op-routable #4: Phase 4 Catalog #1265 contest-equivalence gate verification (Phase 4 deliverable)"
  - "op-routable #5: per-substrate symposium per Catalog #325 (Phase 1+2+3+4 collectively constitute the 6-step contract; Phase 4 lands the empirical anchor)"
council_predicted_mission_contribution: frontier_breaking_enabler
horizon_class: frontier_pursuit
deferred_substrate_id: atw_v2_cooperative_receiver_v2
related_deliberation_ids:
  - path_3_h_atw_v2_cooperative_receiver_cargo_cult_audit_of_existing_scaffold_20260526
  - path_3_h_atw_v2_cooperative_receiver_substrate_design_decision_20260526
  - path_3_h_atw_v2_cooperative_receiver_substrate_design_20260526
score_claim: false
promotion_eligible: false
research_only: true
dispatch_enabled: false
audit_evidence_tag: "[macOS-MLX research-signal]"
---

# Path 3 candidate H — ATW V2 cooperative-receiver L0 SCAFFOLD landed

**Lane:** `lane_path_3_h_atw_v2_cooperative_receiver_cargo_cult_first_20260526` L1
**Subagent:** `path_3_h_atw_v2_cooperative_receiver_cargo_cult_first_20260526`
**Cost:** $0 (entire Phase 1+2+3 audit-design-scaffold cycle)
**Wall-clock:** ~3h total (Phase 1 ~60min + Phase 2 ~25min + Phase 3 ~90min + landing ~15min)

## 1. Phase 1+2+3 deliverables landed

| Phase | Artifact | Commit |
|---|---|---|
| Phase 1 audit | `.omx/research/path_3_h_atw_v2_cooperative_receiver_cargo_cult_audit_of_existing_scaffold_20260526.md` | `06ea98483` |
| Phase 2 decision | `.omx/research/path_3_h_atw_v2_cooperative_receiver_substrate_design_decision_20260526.md` | `683878854` |
| Phase 3 design memo | `.omx/research/path_3_h_atw_v2_cooperative_receiver_substrate_design_20260526.md` | (this batch) |
| Phase 3 L0 scaffold | `src/tac/substrates/atw_v2_cooperative_receiver_v2/` (8 files) | (this batch) |
| Phase 3 smoke trainer | `experiments/train_substrate_atw_v2_cooperative_receiver_v2.py` | (this batch) |
| Phase 3 landing memo | `.omx/research/path_3_h_atw_v2_cooperative_receiver_L0_scaffold_landed_20260526.md` (THIS) | (this batch) |

## 2. L0 SCAFFOLD package contents

```
src/tac/substrates/atw_v2_cooperative_receiver_v2/
├── __init__.py                # 200+ LOC; Catalog #241 contract + Catalog #124 8 fields
├── mlx_renderer.py            # 280+ LOC; MLX-native encoder + decoder + scorer-conditioning
├── numpy_reference.py         # 360+ LOC; numpy reference for FORWARD/INFLATE path (Axis 3)
├── _torch_compat_reference.py # 140+ LOC; PyTorch parity reference (Axis 2 drift verification)
├── _training_only.py          # 100+ LOC; PyTorch training loss skeleton (Catalog #240(c) NotImplementedError)
├── archive.py                 # 230+ LOC; NEW ATWv2CR2 grammar (8 sections; NO dead bytes)
├── inflate.py                 # 130+ LOC; pure-numpy + canonical select_inflate_device
├── registered_substrate.py    # 50+ LOC; Catalog #241 + #240(c) registration
└── tests/
    ├── __init__.py
    └── test_basic.py          # 350+ LOC; 47 tests covering 8 categories
```

Plus:
- `experiments/train_substrate_atw_v2_cooperative_receiver_v2.py` — MLX smoke trainer

## 3. 3-axis evidence summary (per operator binding directive #3 NEW)

### Axis 1 — Math + scientific + engineering rigor

- **Atick-Redlich 1990** = SINGLE substrate-optimal anchor per Phase 2 Layer 1 META-unwind (cooperative-receiver loss against fixed receiver = contest SegNet+PoseNet)
- **Tishby-Zaslavsky 2015** = DEMOTED to advisory cross-check (multi-task IB extension; not in core loss)
- **Wyner-Ziv 1976** = DROPPED (preconditions violated: scorer_class_prior_table shared both-sides; reframed as conditional source coding R(D|Y))
- **Ballard 2007** + **Catalog #311** = ego-motion FOE projection conditioning variable (NEW; replaces D4-falsified per-class softmax)
- **Schmidhuber 2009** + **Cover-Thomas 2006** = advisory cross-checks
- **HNeRV** (Chen et al. 2023 arXiv:2304.02633) = canonical decoder pattern
- Per-layer canonical-vs-unique decision table per Catalog #290 (Phase 3 §1; 5 ADOPT + 9 FORK)

### Axis 2 — MLX drift minimization per primitive

- Per-primitive drift bounds documented per pr95 MLX research (drift_determinism + pytorch_drift_mitigation + full_decoder_downstream_scorer)
- `mlx_renderer.py` uses `precise=True` where applicable + per-pair independence (no cross-pair recurrence) to prevent drift compounding
- test_basic.py includes MLX↔PyTorch drift verification for ego-motion FOE projection within Catalog #1265 gate threshold (1e-3)
- End-to-end full-decoder + scorer drift bound 1e-3 to 1e-2 per pr95_mlx_full_decoder_downstream_scorer_drift_landed (acceptable for research-signal; Catalog #1265 gate WITH mitigations)

### Axis 3 — Portability via numpy

- `numpy_reference.py` provides bit-exact-per-primitive reference for FORWARD/INFLATE path (Linear, Conv2d, ReLU, sigmoid, softmax, mean, ego-motion FOE projection)
- `inflate.py` is pure-numpy + canonical select_inflate_device (PyTorch only for the canonical helper; numpy for all parse + load + reconstruction logic)
- `archive.py` is pure-numpy + struct (trivially portable)
- `_training_only.py` is EXPLICITLY non-portable (PyTorch only; documented exception per operator directive #3 acceptance); ISOLATED in its own module
- Forward path + inflate path FULLY operational on CPU-only systems without MLX

## 4. Empirical verification

- **47/47 tests pass**: `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/substrates/atw_v2_cooperative_receiver_v2/tests/test_basic.py` → 47 passed in 0.54s
- **Smoke trainer end-to-end**: `experiments/train_substrate_atw_v2_cooperative_receiver_v2.py --smoke --seed 42 --batch-size 2 --num-pairs 4` → DONE; rgb_0/rgb_1 shapes correct; sigmoid output in [0, 1]
- **Substrate identity constants importable**: `tac.substrates.atw_v2_cooperative_receiver_v2.SUBSTRATE_ID == "atw_v2_cooperative_receiver_v2"` + canonical META unwind constants exposed
- **Byte-mutation tests verify operational consumption**: per_pair_latent_blob + ego_motion_proj_blob byte mutations change parse result (Catalog #139/#272 satisfied)
- **cdf_table_blob extincted**: `"cdf_table_blob" not in PARSER_SECTION_ROLES` (Phase 1 CC-8 unwind operationalized)

## 5. 6-hook wire-in declaration (Catalog #125 NON-NEGOTIABLE)

| Hook | Status at L0 SCAFFOLD | Phase 4 plan |
|---|---|---|
| 1. Sensitivity-map | STUB in __init__.py constants (no module yet) | NEW `tac.sensitivity_map.atw_v2_cooperative_receiver_ego_motion_conditioning` |
| 2. Pareto constraint | DECLARED in Phase 3 design memo | NEW `tac.pareto.atw_v2_cooperative_receiver_ego_motion_conditional_rate` |
| 3. Bit-allocator | DECLARED in Phase 3 design memo | NEW `bit_allocator.atw_v2_cooperative_receiver_ego_motion_v1` |
| 4. Cathedral autopilot dispatch | NOT YET WIRED (recipe to be created Phase 4 with `dispatch_enabled: false`) | NEW `.omx/operator_authorize_recipes/substrate_atw_v2_cooperative_receiver_v2_modal_t4_dispatch.yaml` |
| 5. Continual-learning posterior update | THIS LANDING MEMO + Phase 1 + Phase 2 + Phase 3 frontmatter emit council anchors | `tac.council_continual_learning.append_council_anchor` (deferred to Phase 4 operational mechanism per current frontmatter v2 emission) |
| 6. Probe-disambiguator | DECLARED in Phase 3 design memo | NEW `tools/probe_atw_v2_cooperative_receiver_ego_motion_conditioning_disambiguator.py` |

## 6. Sister coordination (Catalog #230) — current state at landing

**LANDED concurrent to my work**:
- B' Z7-Mamba-2 Phase 1 audit + Phase 2 decision + L0 SCAFFOLD landed (commit `f7b94faa3` series)
- C' NSCS06 v8 chroma_lut Phase 1 audit + Phase 2 decision landed (commit `bac0ec05d` Phase 2; Phase 3 scaffold landed separately)
- Sister R1 recursive adversarial review concurrent (reads not writes)

**NO file-overlap with sister work**: My scope is the NEW substrate package `src/tac/substrates/atw_v2_cooperative_receiver_v2/` + NEW research memos `.omx/research/path_3_h_atw_v2_cooperative_receiver_*` + NEW smoke trainer `experiments/train_substrate_atw_v2_cooperative_receiver_v2.py`. Catalog #340 sister-checkpoint guard PROCEED at every commit.

**Sister convergence patterns**:
- B' + this work both treat MLX-first as DESIGN-ITERATION-SCOPE only; CUDA-paid empirical anchor only at promotion gate per Catalog #1265 + #325
- C' + this work both invoke canonical equation #26 routing decisions (C' IN-DOMAIN `nscs06_v8_chroma_lut` VERIFIED CORRECT; THIS work confirms the EXCLUDED `direct_byte_substitution_on_decode_opaque_raw_sections` context as the structural fix for v1 cdf_table_blob bug class)

## 7. Operator-routable next steps (Phase 4)

1. **D4-equivalent probe**: `tools/probe_atw_v2_cooperative_receiver_ego_motion_conditioning_disambiguator.py` — measures `I(latent; Y_ego_motion)` on a representative archive; CANONICAL DISPATCH BLOCKER per Catalog #313 sister pattern (sister to V1's D4 probe on A1 latents).
2. **PyTorch training loss**: `_training_only.py` skeleton replaced with full forward+backward via canonical `tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss` + score_pair_components per Catalog #164.
3. **MLX→PyTorch export bridge**: per #1251 canonical pattern (sister to Z6 mlx_export_bridge.py) so MLX-trained weights can be promoted via PyTorch archive build.
4. **Catalog #1265 contest-equivalence gate**: `tools/gate_mlx_candidate_contest_equivalence.py` verification on MLX-trained substrate.
5. **Per-substrate symposium evidence** (Catalog #325): Phase 1 audit + Phase 2 decision + Phase 3 design + Phase 4 empirical anchor collectively constitute the canonical 6-step contract.
6. **`_full_main` NotImplementedError lift**: Phase 4 council approval per Catalog #240(c); ratifies the substrate is at OPTIMAL FORM per Catalog #315 before paid Modal/Lightning/Vast.ai empirical dispatch.

## 8. Cross-references

- Phase 1 audit memo: `.omx/research/path_3_h_atw_v2_cooperative_receiver_cargo_cult_audit_of_existing_scaffold_20260526.md` (commit `06ea98483`)
- Phase 2 design decision: `.omx/research/path_3_h_atw_v2_cooperative_receiver_substrate_design_decision_20260526.md` (commit `683878854`)
- Phase 3 design memo: `.omx/research/path_3_h_atw_v2_cooperative_receiver_substrate_design_20260526.md` (this batch)
- v1 falsification anchor: `.omx/research/atw_v2_cdf_table_blob_reconciliation_codex_byte_mutation_smoke_falsified_20260521.md` (commit `057130de4` empirical evidence preserved)
- Sister B' Z7-Mamba-2 L0 SCAFFOLD landed: `.omx/research/path_3_b_z7_mamba_2_L0_scaffold_landed_20260526.md`
- Sister C' NSCS06 v8 chroma_lut Phase 2: `.omx/research/path_3_c_nscs06_v8_chroma_lut_substrate_design_decision_20260526.md`
- Canonical Z6 mlx_renderer pattern: `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py`
- Canonical pr95_hnerv_mlx pattern: `src/tac/local_acceleration/pr95_hnerv_mlx.py`
- Canonical equation #26 EXCLUDED context: `.omx/state/canonical_equations_registry.jsonl` (`direct_byte_substitution_on_decode_opaque_raw_sections`)

## 9. APPEND-ONLY footer per Catalog #110/#113

This landing memo is APPEND-only. Phase 1 audit + Phase 2 design decision + Phase 3 design memo + L0 SCAFFOLD package files + smoke trainer + all predecessor memos are PRESERVED. The L0 SCAFFOLD operationalizes the Phase 2 Layer 1 META-unwind (Atick-Redlich as SINGLE substrate-optimal anchor + ego-motion FOE projection conditioning surface) per the cargo-cult-pass-first methodology + 3-axis evidence discipline per operator binding directive #3.

**Council verdict**: T2 PROCEED (12 attendees; sextet + 6 topical specialists; AssumptionAdversary HARD-EARNED-EMPIRICALLY-VERIFIED on Path (b) operationalization).

**Mission contribution per Catalog #300**: `frontier_breaking_enabler` (L0 SCAFFOLD landed; substrate is now an actuator-ready candidate for Phase 4 D4-equivalent probe + paid CUDA dispatch consideration via Catalog #1265 gate).

**Lane**: `lane_path_3_h_atw_v2_cooperative_receiver_cargo_cult_first_20260526` L1 (impl_complete: Phase 1 audit + Phase 2 decision + Phase 3 design + L0 SCAFFOLD + smoke trainer + landing memo).
