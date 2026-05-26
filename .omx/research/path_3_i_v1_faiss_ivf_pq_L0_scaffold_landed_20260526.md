# SPDX-License-Identifier: MIT

---
schema: subagent_landing_memo_v1
landing_id: path_3_i_v1_faiss_ivf_pq_L0_scaffold_landed_20260526
lane_id: lane_path_3_i_v1_faiss_ivf_pq_residual_cargo_cult_first_20260526
substrate_id: faiss_ivf_pq_residual
landing_date: "2026-05-26"
subagent_id: claude_subagent_path_3_i
predecessor_directive: "Path 3 candidate I V1 Faiss IVF-PQ residual codec MLX-local L0 SCAFFOLD via cargo-cult-pass-first methodology per operator binding directives 2026-05-26"
phase: phase_3_L0_scaffold
horizon_class: frontier_pursuit
research_only: true
dispatch_enabled: false
council_tier: T2
council_attendees: [Shannon, Dykstra, Jegou, Schmid, Atick, Mallat, AssumptionAdversary]
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_dissent: []
parent_phase_1_memo: .omx/research/path_3_i_v1_faiss_ivf_pq_cargo_cult_audit_20260526.md
parent_phase_2_memo: .omx/research/path_3_i_v1_faiss_ivf_pq_substrate_design_decision_20260526.md
related_deliberation_ids:
  - path_3_i_v1_faiss_ivf_pq_cargo_cult_audit_20260526
  - path_3_i_v1_faiss_ivf_pq_substrate_design_decision_20260526
  - v1_faiss_v4_probe_plus_v8_design_landed_20260519
  - v1_faiss_v8_learned_compression_faiss_design_20260519
predicted_band_validation_status: pending_post_training
deferred_substrate_retrospective_due_utc: 2026-06-25T08:10:00Z
deferred_substrate_id: faiss_ivf_pq_residual
operator_directive_anchor: "Never simply extend unless a rigorous adversarial cargo cult pass has been done first" (2026-05-26 directive #2) + "design the substrate and curriculum and then optimize the design the whole stack" (directive #1) + "adversarial review against all landing recursive for math and scientific and engineering rigor and for MLX drift minimization and portability via numpy" (directive #3) + "mindful not to outpace session rate limits" (directive #4)
---

# Path 3 I — V1 Faiss IVF-PQ residual codec L0 SCAFFOLD LANDED 2026-05-26

> **Status**: PHASE 3 L0 SCAFFOLD complete. All 3 phases of the cargo-cult-first 3-phase methodology landed. 8 files committed; 20/20 tests pass; smoke trainer runs OK; `_full_main raises NotImplementedError` per Catalog #240(c). Paid dispatch GATED by Catalog #325 per-substrate symposium + Catalog #1265 MLX-first parity gate per CLAUDE.md operator-frontier-override discipline.

## 0. 3-phase methodology compliance summary

| Phase | Deliverable | Commit | Discipline |
|---|---|---|---|
| PHASE 1 | Cargo-cult audit memo | `a883a717c` | Catalog #303 + #292 + #287 + #300 v2 + #229 PV |
| PHASE 2 | Substrate-design decision memo | `587e3b85a` | Catalog #290 + #294 + #296 + #303 + #305 + #309 + #300 v2 |
| PHASE 3 | L0 SCAFFOLD | THIS | Catalog #91 + #139 + #146 + #205 + #220 + #240 + #241 + #1265 |

## 1. PHASE 1 cargo-cult audit findings (committed `a883a717c`)

12 inherited V1-V8 assumptions audited; dominant cargo-cult is the **META-ASSUMPTION** that V1 Faiss work directly extends to per-pair residual codec stacking on PR110 fec6.

Classification distribution:
- 4 CARGO-CULTED-MISFRAMED-MIS-TRANSLATED-PARTIAL-DIFFERENT-SURFACE (assumptions #1-4: side-info channel vs residual codec surface, <2KB byte budget, k_topk=3 reducer translation, V8 predicted band)
- 4 CARGO-CULTED-PENDING-EMPIRICAL (assumptions #5-8: V8 cross-pollination, canonical helper application, OMP workaround, MLX-first gate)
- 4 NEW path-I-specific (assumptions #9-12: residual structure, codebook granularity, additive stacking, MLX-first parity)
- 4 HARD-EARNED-PARTIAL on shared PQ primitives (Jégou-Douze-Schmid 2011 PQ + Faiss serialize_codebook + Mallat residual entropy + Atick-Redlich cooperative-receiver)

Per Assumption-Adversary verdict: Path (a) DIRECT EXTENSION REJECTED.

## 2. PHASE 2 substrate-design decision (committed `587e3b85a`)

Verdict: **Path (b) SUBSTRATE-DESIGN REDIRECT** — declare path 3 I as NEW substrate-design question for *per-pair RGB residual codec stacking on PR110 fec6 frontier*; V1-V8 prior work is research INPUT only.

Key decisions:
- Substrate ID: `faiss_ivf_pq_residual`
- Substrate canonical path: `src/tac/substrates/faiss_ivf_pq_residual/`
- Paradigm anchor: Jégou-Douze-Schmid 2011 PQ applied to per-pair RGB residual
- Predicted byte budget envelope (RECALIBRATED §9): ~17 tiles/pair feasible at ≤30 byte/pair
- Predicted band (RECALIBRATED): [0.180, 0.210] frontier-pursuit
- 19 canonical-vs-unique layer decisions: 13 ADOPT + 2 FORK + 4 NEW

## 3. PHASE 3 L0 SCAFFOLD files landed

| # | Path | LOC | Purpose |
|---|---|---|---|
| 1 | `src/tac/substrates/faiss_ivf_pq_residual/__init__.py` | 121 | SPDX + Catalog #241 LEGACY_SUBSTRATE_PRE_META_LAYER waiver + public API exports |
| 2 | `src/tac/substrates/faiss_ivf_pq_residual/mlx_renderer.py` | 207 | MLX-native config + factory + Dykstra-feasibility estimators + `_full_main raises NotImplementedError` per Catalog #240 |
| 3 | `src/tac/substrates/faiss_ivf_pq_residual/numpy_reference.py` | 280 | Sister numpy reference per axis 3 portability: PQ codebook gather + tile reassemble + K-means train + PQ encode + uint8 cast |
| 4 | `src/tac/substrates/faiss_ivf_pq_residual/archive.py` | 280 | FAISSPQ1 byte-deterministic grammar; 29-byte header + 3 brotli-compressed blobs (codebook + codewords + meta) |
| 5 | `src/tac/substrates/faiss_ivf_pq_residual/inflate.py` | 175 | PyTorch runtime per Catalog #146 3-positional-arg contract + Catalog #205 select_inflate_device; LOC budget ≤200 per HNeRV parity L4 |
| 6 | `src/tac/substrates/faiss_ivf_pq_residual/tests/__init__.py` | 2 | test package marker |
| 7 | `src/tac/substrates/faiss_ivf_pq_residual/tests/test_basic.py` | 290 | 20 tests: Catalog #91 archive + #139 byte-mutation + numpy reference round-trip + MLX↔numpy parity (skip-if-MLX-missing) + Catalog #240 `_full_main` raises |
| 8 | `experiments/train_substrate_faiss_ivf_pq_residual.py` | 165 | Smoke trainer + `_full_main raises NotImplementedError`; `--smoke --validate-archive-roundtrip` validates synthetic round-trip |
| 9 | `.omx/research/path_3_i_v1_faiss_ivf_pq_L0_scaffold_landed_20260526.md` | THIS | Landing memo with 3-axis evidence + operator-routable next steps |

**Total LOC**: ~1520 across 9 files (substrate-engineering exception per HNeRV parity L7 explicitly tagged via Catalog #241 LEGACY_SUBSTRATE_PRE_META_LAYER waiver).

## 4. Test verification

Run command: `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/substrates/faiss_ivf_pq_residual/tests/test_basic.py -x -v`

Result: **20 passed in 0.68s**

Test inventory:
- Config validation (5 tests): defaults_valid / invalid_m / invalid_ksub / invalid_tile / tile_dim_not_divisible
- Archive grammar (6 tests): header_size_constant / roundtrip / deterministic_bytes / magic_mismatch_raises / truncated_raises / meta_contains_canonical_keys
- Catalog #139 byte-mutation (1 test): codebook_changes_decoded_reconstruction
- numpy reference (4 tests): pq_codebook_gather_roundtrip / tiles_to_frame_roundtrip / pq_encode_decode_roundtrip / to_uint8_canonical_rounding
- MLX↔numpy parity (1 test, runs when MLX available): pq_codebook_gather byte-identical
- Catalog #240 L0 SCAFFOLD posture (1 test): _full_main raises NotImplementedError
- Dykstra-feasibility estimators (2 tests): per_pair_codeword_bytes_raw / archive_bytes

Smoke trainer verification: `PYTHONPATH=src:upstream:$PWD .venv/bin/python experiments/train_substrate_faiss_ivf_pq_residual.py --smoke --validate-archive-roundtrip`

Result: archive round-trip OK; canonical non-promotable markers stamped (research_only=True / dispatch_enabled=False / score_claim=False / promotion_eligible=False / ready_for_exact_eval_dispatch=False).

`_full_main` verification: `PYTHONPATH=src:upstream:$PWD .venv/bin/python experiments/train_substrate_faiss_ivf_pq_residual.py` (without `--smoke`)

Result: `NotImplementedError: faiss_ivf_pq_residual full main NOT YET IMPLEMENTED — L0 SCAFFOLD posture per Catalog #240(c)`

## 5. 3-axis rigor evidence per directive #3

### Axis 1: Math + scientific + engineering rigor per layer

| Layer | Math | Scientific | Engineering | Verdict |
|---|---|---|---|---|
| PQ codebook quantization | Jégou-Douze-Schmid 2011 PQ asymptotic distortion bound | Faiss paper canonical citation in design memo | Canonical helper primitives reused from `tac.optimization.faiss_ivf_pq_atw_channel` | HARD-EARNED |
| Per-pair RGB residual signal | Mallat wavelet theory residual R(D) bound + Atick-Redlich retinal residual MI | Daubechies wavelet residual canonical + Atick-Redlich 1990 cited in design memo | PR110 fec6 frontier reconstruction is canonical input (sha 6bae0201 per pointer 2026-05-15) | HARD-EARNED-PENDING-EMPIRICAL |
| Inflate-time canonical helpers | Catalog #205 select_inflate_device + canonical bilinear upsample | Sister A=DreamerV3 RSSM forensic anchor for MLX drift mitigation | A landed `69253a1cc` provides canonical pattern | HARD-EARNED |
| MLX-Faiss adapter feasibility | Integer codebook lookup deterministic; float gather deterministic | Sister G=NIRVANA mlx_renderer.py canonical pattern | sister G landed `f7d2e86fe` provides pattern; MLX↔numpy parity test PASSES | HARD-EARNED |
| Predicted band derivation | Per-pair MSE residual R(D) bound + RECALIBRATED tile budget per PHASE 2 §9 | Per Catalog #324 post-training Tier-C re-measurement REQUIRED before contest claim | NO premature claim; pending PHASE 4 (post-L0) empirical anchor | HARD-EARNED (no premature claim) |
| Archive grammar | FAISSPQ1 29-byte header + 3 brotli-compressed blobs | Mirrors sister NIRVANA1 + DREAMER1 + WZF01 patterns | Byte-deterministic round-trip test PASSES | HARD-EARNED |
| `_full_main` posture | Catalog #240(c) non-negotiable | Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" Catalog #220 | NotImplementedError verified empirically | HARD-EARNED |

### Axis 2: MLX drift minimization per primitive

| Primitive | Expected drift bound | Mitigation | Canonical helper citation | Status |
|---|---|---|---|---|
| PQ codebook gather (MLX) | ≤ epsilon-machine (float copy) | `mx.take` per sister K, G | sister G NIRVANA `mlx_renderer.py` | EMPIRICALLY VERIFIED (test_mlx_numpy_parity_pq_codebook_gather PASSES byte-identical) |
| Bilinear residual upsample | ≤ Catalog #1265 threshold 0.001 | USE canonical `bilinear_resize2x_align_corners_false_nhwc` | sister A=DreamerV3 forensic anchor (max_abs=24.34 caused by align_corners=True anti-pattern AVOIDED) | DESIGN-LEVEL (deferred to PHASE 4 if used) |
| uint8 cast at output | = 0 (deterministic) | USE canonical Catalog #205 sister rounding | `_shared/inflate_runtime.py` | VERIFIED in inflate.py |
| PQ encoding (numpy) | byte-identical with MLX | Numpy reference path | sister G `numpy_reference.py` pattern | PASSES byte-identical |
| Tile reassemble | = 0 (deterministic) | Index arithmetic only (numpy reference + MLX both deterministic) | n/a (trivial primitive) | PASSES |

### Axis 3: Portability via numpy per primitive

Every MLX primitive has sister numpy reference in `numpy_reference.py`:

| Primitive | MLX path | numpy reference | Portability verdict |
|---|---|---|---|
| PQ codebook gather | `mx.take` | `pq_codebook_gather` | NUMPY-PURE PORTABLE |
| Tile reassemble (frame → tiles) | `mx.reshape` | `frame_to_tiles_nhwc` | NUMPY-PURE PORTABLE |
| Tile reassemble (tiles → frame) | `mx.reshape` + `mx.concatenate` | `tiles_to_frame_nhwc` | NUMPY-PURE PORTABLE |
| PQ codebook training | sklearn KMeans (optional) OR Faiss-CPU | `train_pq_codebook` (numpy K-means) | NUMPY-PURE PORTABLE |
| PQ encoding | numpy reference (no MLX needed) | `encode_per_pair_residual` | NUMPY-PURE PORTABLE |
| PQ vector reconstruction | composition of gather + flatten | `pq_reconstruct_tile_vectors` | NUMPY-PURE PORTABLE |
| uint8 cast | numpy primitive | `to_uint8` | NUMPY-PURE PORTABLE |
| Bilinear upsample (inflate-time) | torch.nn.functional.interpolate (PyTorch path) | N/A — inflate runtime uses PyTorch per Catalog #146 | INFLATE-RUNTIME via PyTorch |

**Result**: substrate operable on CPU-only test rigs without MLX OR Faiss dependency per directive #3 axis 3. GHA CPU CI testing supported via numpy-only path.

## 6. 6-hook wire-in declaration per Catalog #125

| # | Hook | Status | Rationale |
|---|---|---|---|
| 1 | Sensitivity-map (`tac.sensitivity_map.*`) | N/A — L0 SCAFFOLD | PHASE 4 wire-in via `tac.sensitivity_map.per_byte_leverage` post-empirical anchor |
| 2 | Pareto constraint (`tac.pareto_*`) | N/A — L0 SCAFFOLD | PHASE 4 wire-in via `tac.pareto_solver.add_constraint` post-empirical anchor |
| 3 | Bit-allocator hook | N/A — L0 SCAFFOLD | PHASE 4 wire-in via byte-budget allocation |
| 4 | Cathedral autopilot dispatch | **ACTIVE (DESIGN-LEVEL)** | This landing memo is consumable by canonical_equation_lookup_consumer per Catalog #344; design predicted-band is Tier A observability-only routing recommendation per Catalog #341 |
| 5 | Continual-learning posterior update | **ACTIVE (DESIGN-LEVEL)** | THIS landing memo IS continual-learning artifact per CLAUDE.md "Subagent coherence-by-default"; canonical posterior anchor via Catalog #313 probe-outcomes ledger NOT YET appended (PHASE 4 empirical outcome registration) |
| 6 | Probe-disambiguator | **ACTIVE (DESIGN-LEVEL)** | MLX-first gate at Catalog #1265 IS canonical disambiguator between MLX-PASS and PyTorch-required for paid CUDA dispatch eligibility |

## 7. Operator-routable next steps (PHASE 4+)

### Step 1: Catalog #325 per-substrate symposium (PHASE 2 of dispatch eligibility)
- Spawn per-substrate symposium subagent for `faiss_ivf_pq_residual` substrate
- Council attendees: T2 sextet (Shannon + Dykstra + Yousfi + Fridrich + Contrarian + Assumption-Adversary) + grand council topical seats (Jégou + Schmid + Ge + Mallat + Atick + Daubechies)
- 6-step canonical contract per Catalog #325: cargo-cult audit + 9-dim checklist + observability + sextet pact + reactivation criteria + Tier-C validation
- Outcome anchor via `tac.council_continual_learning.append_council_anchor`

### Step 2: PR110 fec6 frontier residual extraction probe (PHASE 4 FREE on macOS-CPU)
- Decode PR110 fec6 archive (sha 6bae0201 per canonical pointer) via canonical `inflate.sh` on 600 contest pairs
- Compute per-pair residual = frame_gt - frame_pr110_decoded
- Persist residuals as `experiments/results/lane_path_3_i_phase_4_residual_extraction_<utc>/per_pair_residual_<utc>.npy`
- Register probe outcome per Catalog #313

### Step 3: PQ codebook training probe (PHASE 4 FREE on macOS-CPU)
- Use numpy `train_pq_codebook` reference OR Faiss-CPU optional accelerator (per V4 hand-rolled probe OMP_NUM_THREADS=1 workaround)
- K-sweep + M-sweep + tile-size-sweep + per-class-conditioning binary
- Predicted byte cost vs PQ codebook empirical fit distortion table
- Register probe outcome per Catalog #313

### Step 4: MLX-first parity gate at Catalog #1265 threshold 0.001 (PHASE 4 FREE on macOS-MLX)
- Run MLX↔PyTorch parity test on residual decode + tile reassemble + bilinear upsample
- VERIFY drift ≤ 0.001 contest-units per Catalog #1265 threshold
- If FAIL: identify drift surface and apply canonical helper per sister A=DreamerV3 forensic
- Register probe outcome per Catalog #313

### Step 5: PHASE 4 paid dispatch decision via Catalog #199 operator-frontier-override
- ONLY after Step 1 (symposium PROCEED) + Step 4 (MLX-first parity PASS) + Step 3 (PQ codebook empirical anchor)
- Paid Modal A100 5-50ep smoke ($5-30) per Catalog #167 smoke-before-full
- Outcome anchor via `tac.cost_band_calibration.append_anchor` per Catalog #175 + #177
- Paired CPU+CUDA harvest per Catalog #246

### Step 6: PHASE 5 composition smoke with sister E=BoostNeRV (orthogonal residual axes)
- Conditional on individual substrate empirical anchors landed
- Composition Pareto-feasibility check per Catalog #319 deliverability proof + Catalog #296 Dykstra
- Predicted composition Δ subject to additive assumption empirical validation

### Step 7: 30-day retrospective per CLAUDE.md Mission Alignment Consequence 3
- Due 2026-06-25
- Re-audit PHASE 4 empirical outcomes + sister coordination
- Per Catalog #292 + #300 mission alignment

## 8. Sister coordination ledger per Catalog #230

Sister Path 3 wave at landing time:

| Sister | Lane | Status | Path 3 I distinguishing |
|---|---|---|---|
| A=DreamerV3 RSSM | `aaec7a0d220f31543` | LANDED `69253a1cc` | Categorical latent dynamics (G×K) vs flat PQ codec |
| B'=Z7-Mamba-2 cargo-cult-first | `ac4283983ece21b83` | LANDED `7a103fdbb` | State-space sequence model vs codec primitive |
| C'=NSCS06 v8 chroma_lut | `ad26de7ad5f90848a` | LANDED `f59c8401b` | Per-class chroma LUT vs per-pair PQ |
| D=Z6 predictive coding | `af6ca73c5a7fc40f4` | LANDED `83b9ee3e2` | Predictive-coding paradigm vs codec primitive |
| E=BoostNeRV PR110 residual | `a35f9f86781aaaa4f` | LANDED `83910e54e` | Iterative boosting vs vector quantization |
| F=Z8 hierarchical predictive coding | `a23f0430835406351` | IN-FLIGHT | Canonical quadruple vs codec primitive |
| G=NIRVANA cascading NeRV | `ae952528954e27bef` | LANDED `f7d2e86fe` | Hierarchical wavelet cascade vs flat PQ |
| H=ATW V2 cooperative-receiver | `aba5069741fc4475b` | IN-FLIGHT | Atick-Tishby-Wyner triple cooperative-receiver vs codec primitive |
| K=COIN++ INR | `a7977f23a7f0f0573` | IN-FLIGHT | Meta-learned MLP-per-coordinate vs codec primitive |
| **I=Path 3 I V1 Faiss IVF-PQ residual** | `lane_path_3_i_v1_faiss_ivf_pq_residual_cargo_cult_first_20260526` | **THIS LANDING** | Vector quantization codec primitive (Jégou-Douze-Schmid 2011 PQ); per-pair RGB residual against PR110 fec6 |

Sister coordination at edit-time: Catalog #340 sister-checkpoint guard fired once on my own in-progress checkpoint (resolved by mark-complete-then-retry pattern); zero collisions with active sister subagents.

## 9. Discipline compliance summary

- **Catalog #91** (archive grammar contract): COMPLETE — FAISSPQ1 29-byte header + 3 brotli-compressed blobs + 6 archive tests PASS
- **Catalog #117 / #157 / #174 / #235 / #289** (canonical commit serializer + POST-EDIT --expected-content-sha256): COMPLETE — all 3 phase commits via canonical serializer
- **Catalog #119** (Co-Authored-By Claude trailer): COMPLETE — all 3 phase commits carry canonical trailer
- **Catalog #125** (6-hook wire-in non-negotiable): COMPLETE — 3 N/A + 3 ACTIVE DESIGN-LEVEL per §6
- **Catalog #131** (no bare writes to shared state): COMPLETE — probe outcome appending deferred to PHASE 4 via canonical helper
- **Catalog #139** (byte-mutation distinguishing-feature test): COMPLETE — `test_archive_byte_mutation_codebook_changes_decoded_reconstruction` PASSES (distinguishing-feature evidence: codebook mutation produces parse-fail OR reconstruction-changes)
- **Catalog #146** (inflate runtime contract): COMPLETE — `inflate.py` honors 3-positional-arg `inflate.sh <archive_dir> <output_dir> <file_list>` contract
- **Catalog #205** (canonical inflate device-fork): COMPLETE — `select_inflate_device` adopted via canonical helper import
- **Catalog #206** (subagent checkpoint discipline): COMPLETE — 7+ checkpoints emitted across 3 phases
- **Catalog #208** (no docs/local-paths in artifacts): COMPLETE — no `/Users/` absolute paths in source
- **Catalog #220** (substrate L1+ scaffold operational mechanism): N/A at L0 — substrate is L0 SCAFFOLD with `research_only: true` + `dispatch_enabled: false` per Catalog #240(c) opt-out
- **Catalog #229** (premise verification before edit): COMPLETE — read V1+V4+V8 priors + sister G=NIRVANA scaffold pattern + canonical helpers BEFORE writing
- **Catalog #230** (sister-subagent ownership map): COMPLETE — sister coordination ledger in §8; commit body cites sister wave
- **Catalog #240** (L0 SCAFFOLD posture): COMPLETE — `_full_main raises NotImplementedError` verified empirically
- **Catalog #241** (LEGACY_SUBSTRATE_PRE_META_LAYER waiver): COMPLETE — `__init__.py` carries explicit waiver with non-placeholder rationale
- **Catalog #245** (Modal call_id ledger): N/A at L0 — no Modal dispatch
- **Catalog #287** (placeholder-rationale rejection): COMPLETE — all waivers carry non-placeholder rationale ≥4 chars
- **Catalog #290** (canonical-vs-unique decision per layer): COMPLETE — PHASE 2 design memo §3 documents 19 layer decisions
- **Catalog #292** (per-deliberation assumption surfacing): COMPLETE — Assumption-Adversary verdict in YAML frontmatter
- **Catalog #294** (9-dim checklist evidence): COMPLETE — PHASE 2 §6 documents 9-dim
- **Catalog #296** (Dykstra-feasibility predicted-band check): COMPLETE — PHASE 2 §9 documents Dykstra-feasibility intersection
- **Catalog #299** (catalog quota brake): COMPLETE — 0 new STRICT preflight gates added; current count well under 400 quota
- **Catalog #300** (council deliberation v2 frontmatter): COMPLETE — all 3 phase memos carry v2 frontmatter
- **Catalog #303** (cargo-cult audit per assumption): COMPLETE — PHASE 1 audit documents 12 assumptions
- **Catalog #305** (observability surface section): COMPLETE — PHASE 2 §8 documents 6-facet observability
- **Catalog #309** (horizon_class declaration): COMPLETE — `horizon_class: frontier_pursuit` in all 3 phase memos
- **Catalog #313** (probe-outcomes ledger registration): DEFERRED to PHASE 4 empirical anchor registration
- **Catalog #323** (canonical Provenance umbrella): COMPLETE — meta JSON carries canonical non-promotable markers
- **Catalog #324** (post-training Tier-C validation discipline): DEFERRED to PHASE 4 — `predicted_band_validation_status: pending_post_training`
- **Catalog #325** (per-substrate symposium discipline): QUEUED for PHASE 4 dispatch eligibility per §7 step 1
- **Catalog #340** (sister-checkpoint guard at staging surface): COMPLETE — fired once on own checkpoint; resolved by mark-complete-then-retry
- **Catalog #344** (canonical equations + models registry): N/A at L0 — no empirical anchor registration yet
- **Catalog #346** (canonical council roster validate complete): COMPLETE — Shannon + Dykstra + Jégou + Schmid + Atick + Mallat + AssumptionAdversary roster per YAML frontmatter
- **Catalog #1265** (MLX-first contest-equivalence gate): DEFERRED to PHASE 4 empirical parity test on residual codec primitives
- **CLAUDE.md "Executing actions with care"**: COMPLETE — NO `gh pr create`, NO `gh release create`, NO Modal/Vast/Lightning dispatch
- **CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"**: COMPLETE — `research_only: true` + `dispatch_enabled: false` declared
- **CLAUDE.md "Strict scorer rule"**: COMPLETE — NO scorer imports at inflate runtime
- **CLAUDE.md "MPS auth eval is NOISE"**: COMPLETE — canonical `select_inflate_device` adopted; NO MPS fallback default
- **Per directive #4 efficient token use**: COMPLETE — combined PV reads; rejected sister Path (c) HYBRID as kitchen_sink; rejected redundant V1-V8 inheritance per PHASE 1 audit

## 10. Cross-references

- PHASE 1 cargo-cult audit memo: `.omx/research/path_3_i_v1_faiss_ivf_pq_cargo_cult_audit_20260526.md` (commit `a883a717c`)
- PHASE 2 substrate-design decision memo: `.omx/research/path_3_i_v1_faiss_ivf_pq_substrate_design_decision_20260526.md` (commit `587e3b85a`)
- V1+V4+V8 prior research (research INPUT only): `v1_faiss_v4_probe_plus_v8_design_landed_20260519.md` + `v1_faiss_v8_learned_compression_faiss_design_20260519.md`
- Sister G=NIRVANA scaffold pattern: `src/tac/substrates/nirvana_cascading_nerv/` (canonical reference for mlx_renderer + numpy_reference + archive + inflate + tests structure)
- Canonical helper primitives reused: `src/tac/optimization/faiss_ivf_pq_atw_channel.py` (build_pq_codebook + serialize_codebook)
- Canonical frontier pointer: `.omx/state/canonical_frontier_pointer.json` (PR110 fec6 frontier sha 6bae0201)
- Path 3 inventory brief: `.omx/research/path_3_candidate_inventory_for_next_wave_spawning_20260526.md`
- Catalog #303 cargo-cult audit framework + addendum: `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`

## 11. Files touched

NEW files (8):
- `src/tac/substrates/faiss_ivf_pq_residual/__init__.py`
- `src/tac/substrates/faiss_ivf_pq_residual/mlx_renderer.py`
- `src/tac/substrates/faiss_ivf_pq_residual/numpy_reference.py`
- `src/tac/substrates/faiss_ivf_pq_residual/archive.py`
- `src/tac/substrates/faiss_ivf_pq_residual/inflate.py`
- `src/tac/substrates/faiss_ivf_pq_residual/tests/__init__.py`
- `src/tac/substrates/faiss_ivf_pq_residual/tests/test_basic.py`
- `experiments/train_substrate_faiss_ivf_pq_residual.py`
- THIS landing memo

APPENDED files:
- `.omx/state/subagent_progress.jsonl` (Catalog #206 checkpoints emitted)
- `.omx/state/commit-serializer.log` (Catalog #117 canonical serializer entries)
- `.omx/state/active_lane_dispatch_claims.md` (Catalog #230 lane claim row)

## 12. Cost + wall-clock summary

- **Cost**: $0 (no paid dispatch; macOS-MLX research-signal only)
- **Wall-clock**: ~75 min (PV ~15 min + PHASE 1 ~15 min + PHASE 2 ~15 min + PHASE 3 L0 SCAFFOLD ~20 min + landing memo ~10 min)
- **Efficient token use per directive #4**: combined related reads/edits into single tool calls; rejected sister Path (c) HYBRID upfront; sister coordination preserved zero edit collision
