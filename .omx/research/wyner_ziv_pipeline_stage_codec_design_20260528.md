---
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Wyner
  - Atick-Redlich
  - Tishby-memorial
  - PR95Author
council_quorum_met: true
council_verdict: PROCEED
council_dissent:
  - member: Contrarian
    verbatim: |
      The L0 scaffold's encode-decode roundtrip on synthetic bytes proves only the
      primitive contract is honored; it does NOT prove the WZ stage produces
      meaningful score-savings on real PR101/PR106/A1 pre-entropy bytes. The sister
      primitive's prober anchor predicted 0.47 score-savings per substrate based on
      lzma ratio 0.217-0.228 on raw fp16 state_dict. THIS substrate composes the
      primitive into a substrate package but does NOT yet measure the deliverable
      composition factor (Catalog #227 alpha) on real bytes. The L1 trainer MUST
      measure alpha against a verified base substrate before any score-claim
      promotion. The L0 verdict is PROCEED, NOT PROMOTE.
  - member: Assumption-Adversary
    verbatim: |
      The L0 scaffold's smoke uses synthetic pre-entropy bytes where the
      Y-derivable-prefix is GUARANTEED to overlap (the test constructs Y to
      contain a 512-1024 byte prefix of source). On REAL fp16 state_dict bytes
      the Y-derivable-prefix may be near-zero — the byte distributions are
      different. The substrate's predicted ΔS band assumes the prober's
      0.217-0.228 lzma ratio TRANSLATES to Y-derivable-prefix overlap; this
      assumption is HARD-EARNED for state-dict-vs-state-dict overlap (prober
      anchor) but CARGO-CULTED for state-dict-vs-Comma2k19-side-info overlap
      (no empirical anchor yet). The L1 trainer's first smoke MUST measure
      Y-derivable-prefix density on real bytes BEFORE any L2 promotion.
council_assumption_adversary_verdict:
  - assumption: "lzma ratio 0.217-0.228 on raw fp16 state_dict bytes (sister primitive prober)"
    classification: HARD-EARNED
    rationale: "Empirical prober anchor 2026-05-17 at lane_wyner_ziv_deliverability_prober_20260517"
  - assumption: "Y-derivable-prefix overlap density on state-dict-vs-Comma2k19 side-info"
    classification: CARGO-CULTED
    rationale: "Inherited from prober's state-dict-vs-state-dict measurement; no empirical anchor for cross-substrate composition. L1 first-smoke MUST measure."
  - assumption: "Wyner 1976 R(D|Y) achievable rate at the pipeline-stage primitive surface"
    classification: HARD-EARNED
    rationale: "Information-theory bound; Wyner & Ziv 1976 IEEE Trans IT-22(1):1-10 is the canonical theorem"
  - assumption: "decoder-side PoseNet output IS canonical Y at inflate time"
    classification: HARD-EARNED
    rationale: "upstream/modules.py is contest-deterministic; every contest runner loads the same weights"
  - assumption: "MLX training + numpy-portable inflate composes byte-identically"
    classification: CARGO-CULTED
    rationale: "Sister Z6-v2 + DP1 demonstrate the pattern but per-substrate FP16-vs-FP32 + endian + alignment can drift; L1 first-smoke MUST verify"
  - assumption: "additive composition alpha=1.0 with FEC6 0.19205 frontier"
    classification: CARGO-CULTED
    rationale: "Catalog #227 substrate_composition_matrix default; sister probe-disambiguator at tools/probe_wyner_ziv_composition_alpha_disambiguator.py is the canonical empirical arbiter pending L1 paired smoke"
council_decisions_recorded:
  - "op-routable #1: L1 trainer wires MLX training on Comma2k19-derived side_info_y per Catalog #213 canonical helper; first-smoke measures Y-derivable-prefix density on real PR101/PR106/A1 pre-entropy bytes"
  - "op-routable #2: paired CUDA+CPU auth-eval per Catalog #246 BEFORE any score-claim promotion; defer L2 promotion until composition alpha measured per Catalog #227"
  - "op-routable #3: sister probe-disambiguator tools/probe_wyner_ziv_composition_alpha_disambiguator.py reactivation-pending-empirical per primitive landing memo; resolve before L2"
  - "op-routable #4: if first-smoke measures Y-derivable-prefix density <= 1% (per Assumption-Adversary HARD-EARNED critique), DEFER to alternative side_info_source (torch_defaults / math_constants) per primitive taxonomy"
  - "op-routable #5: Wave 2 sister extension to per-pair PoseNet-output side-info — the deepest class-shift route per Catalog #311 Atick-Tishby-Wyner triple; requires PoseNet inflate-time forward + Catalog #320 attestation OR alternative scorer-free Y derivation"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: null
deferred_substrate_retrospective_due_utc: null
deferred_substrate_id: wyner_ziv_pipeline_stage_codec
related_deliberation_ids:
  - wyner_ziv_pipeline_stage_codec_primitive_landed_20260517
  - z6_v2_cargo_cult_unwind_design_20260527
  - d4_wyner_ziv_frame_0_substrate_20260514
  - wyner_ziv_cooperative_receiver_substrate_20260513
  - grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517
horizon_class: asymptotic_pursuit
predicted_band_validation_status: pending_post_training
substrate_alias: wyner_ziv_pipeline_stage_codec
---

# Wyner-Ziv pipeline-stage codec L0 SCAFFOLD design memo

**Substrate id**: `wyner_ziv_pipeline_stage_codec`
**Lane id**: `lane_wyner_ziv_pipeline_stage_codec_l0_scaffold_20260528`
**Landing UTC**: 2026-05-28T02:57:23Z
**HEAD sha (pre-landing)**: ee7dba11f97ad7ab64ff23670740d538a7f02847
**Author**: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
**Author**: Alejandro Peña <adpena@gmail.com>

Per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council
symposium" non-negotiable + Catalog #325 6-step contract + N1 path-5 STRUCTURAL
CEILING REINFORCED 2026-05-28 routing (memo
`.omx/research/n1_path_5_8_seed_bootstrap_k5_landed_20260528T014859Z.md`)
reactivation criterion #3 *"Wyner-Ziv decoder-side PoseNet side-info"*.

## Routing context

This substrate L0 SCAFFOLD lands the **3rd of 3** next-paradigm reactivation
paths from N1 path 5's STRUCTURAL CEILING REINFORCED verdict on the pose-axis
MLX-surrogate ceiling. The two sister paths already landed in adjacent cycles:

* Path 1 (Full-PyTorch-backprop teacher) — operator-deferred per cost envelope
* Path 2 (Z6-v2 cooperative-receiver pose-axis reformulation) — landed
  commit `afa5ba837` 2026-05-27 (Z6-v2 cargo-cult-unwind L0 SCAFFOLD)
* Path 3 (Wyner-Ziv decoder-side PoseNet side-info) — **THIS substrate**

The Wyner-Ziv pipeline-stage codec primitive at `tac.codec.wyner_ziv_layer`
(lane `lane_wyner_ziv_pipeline_stage_codec_primitive_20260517` at L1; 740
LOC + 64 tests; sister landing 2026-05-17) is the canonical implementation
of the Wyner & Ziv 1976 R(D|Y) achievable-rate theorem applied at the
compression-pipeline-stage byte-stream surface. THIS substrate is the
substrate-scope CONSUMER of that primitive per Catalog #335 canonical
cathedral consumer pattern + Catalog #341 Tier A canonical-routing markers.

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" operating mode + Catalog #290:

| Layer | Decision | Rationale |
|---|---|---|
| Identity (id / lane_id / target_modes / deployment_target) | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #241/#242 36-field schema |
| Archive grammar | UNIQUE | `monolithic_single_file_wzpsc01` MAGIC + 25-byte HEADER + sorted-keys JSON meta + (main, side) data section is THIS substrate's mathematical structure (Wyner-Ziv split). Sister substrates use different MAGICs (D1 WZF0, DP1 DP1\\x00, Z6-v2 z6v2cu1). |
| Parser section manifest | UNIQUE | 5 sections (header + main_stream + side_stream + y_derivation_meta + meta) reflect the Wyner-Ziv canonical (main, side, Y) triple per the primitive's API. |
| Inflate runtime LOC budget | UNIQUE (200 substrate-engineering waiver) | HNeRV parity L4 ≤200 with explicit substrate_engineering rationale. The inflate routes through the canonical primitive's reconstruct path; ~110 effective LOC physical + ~100 docstring. |
| Runtime dep closure | UNIQUE (numpy, brotli) | 8th MLX-FIRST + numpy-portable directive. MLX is NOT a runtime dep. lzma + zlib are stdlib; brotli is the only non-stdlib dep. Within HNeRV parity L4 ≤2 deps. |
| Export format | ADOPT_CANONICAL ("custom") | MLX state_dict → npz → ZIP-member → numpy inflate is the canonical 8th MLX-FIRST bridge per CLAUDE.md standing directive. |
| Score-aware loss | ADOPT_CANONICAL ("custom") | Cooperative-receiver MI-max via decoder-side PoseNet side-info per Atick-Redlich 1990. The L1 trainer will bind to `tac.codec.cooperative_receiver.atick_redlich` if available OR inline the rate term. |
| Bolt-on LOC budget | N/A (substrate_engineering) | HNeRV parity L7 substrate-engineering exception. 999 sentinel per Catalog #241 invariant. |
| No-op detector | ADOPT_CANONICAL (True) | Catalog #105/#139 byte-mutation smoke + Catalog #220/#272 operational mechanism. The encode-decode roundtrip IS the canonical no-op detector. |
| Score-improvement mechanism status | UNIQUE (RESEARCH_ONLY at L0) | L0 scaffold; flips to OPERATIONAL at L1 per Catalog #220 invariant + Catalog #241 paired-field rule. |
| Runtime overlay consumed | UNIQUE (False at L0) | Paired with RESEARCH_ONLY per `SubstrateContract.__post_init__`. |
| Recipe smoke/research_only | ADOPT_CANONICAL (both True) | L0 scaffold contract per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable + Catalog #240. |
| Recipe min_smoke_gpu / min_vram_gb | UNIQUE (T4 / 12 GB) | Placeholder ladder; revised at L1 per actual measurement. The primitive itself is CPU-runnable so the smoke ladder is flexible. |
| Recipe video/pyav/canary | ADOPT_CANONICAL | sister Z6-v2 pattern. canary_status=independent_substrate per HNeRV parity L7 (this substrate composes with ANY base substrate's pre-entropy stage). |
| Cost band | UNIQUE (T4 modal 50 ep $0.30) | L0 smoke envelope; revised post-empirical. |
| Hook #1 sensitivity | ADOPT_CANONICAL (scorer_conditional_entropy_map_v1) | Routes through sister Wire-in #2 `tac.sensitivity_map.wyner_ziv_reweight`. |
| Hook #2 Pareto | ADOPT_CANONICAL (rate_distortion_v1) | Wyner 1976 IS the canonical R(D\\|Y) Pareto constraint at the pipeline-stage surface. |
| Hook #3 bit-allocator | ADOPT_CANONICAL (per_tensor_uniform) | L0 default; L1 may FORK to per_channel_lsq if empirical justifies. |
| Hook #4 cathedral autopilot ranker class-shift token | UNIQUE | `pipeline_stage_codec_decoder_side_posenet_side_info_wyner_ziv_1976_per_catalog_311` — this substrate's distinguishing token per Catalog #311 + Catalog #344 canonical equations registry. |
| Hook #5 continual-learning | ADOPT_CANONICAL (macos_cpu_advisory at L0) | MLX-local non-promotable per Catalog #192 + 8th MLX-FIRST directive. L1 paired-smoke may promote to paired_axis. |
| Hook #6 probe-disambiguator | ADOPT_CANONICAL | sister stub `tools/probe_wyner_ziv_composition_alpha_disambiguator.py` (landed 2026-05-17). |
| Catalog compliance declarations | UNIQUE (5 tokens) | 146 + 164 + 205 + 220 + 226 per the substrate's contract obligations. |

## Cargo-cult audit per assumption

Per CLAUDE.md "Substrate design memos MUST include cargo-cult audit"
(Catalog #303 + hard-earned-vs-cargo-culted addendum 2026-05-15):

| Assumption | Classification | Hard-earned source / Unwind path |
|---|---|---|
| Wyner & Ziv 1976 R(D\\|Y) achievable-rate theorem | HARD-EARNED | Wyner & Ziv 1976 IEEE Trans IT-22(1):1-10; canonical information-theory bound. |
| Decoder-side PoseNet output IS canonical Y at inflate time | HARD-EARNED | upstream/modules.py is contest-deterministic; every contest runner loads the same FastViT-T12 weights. |
| Y-derivable-prefix detection IS the simplest faithful WZ split | HARD-EARNED | sister primitive's design docstring + 740-LOC implementation 2026-05-17; tests pin byte-identical roundtrip. |
| Atick-Tishby-Wyner triple covers this substrate's mathematical structure | HARD-EARNED | Catalog #311 grand council roster expansion 2026-05-15 places all 3 seats on cooperative-receiver-and-side-info-coding deliberations. |
| sister primitive's lzma ratio 0.217-0.228 on fp16 state_dict | HARD-EARNED | Empirical prober anchor 2026-05-17 at `lane_wyner_ziv_deliverability_prober_20260517`. |
| **Y-derivable-prefix density on Comma2k19-vs-state-dict** | **CARGO-CULTED** | **Inherited from prober's state-dict-vs-state-dict measurement; cross-substrate Y derivation is unmeasured. Unwind: L1 first-smoke measures Y-derivable-prefix density on real bytes BEFORE L2 promotion.** |
| **additive composition alpha=1.0 with FEC6 frontier** | **CARGO-CULTED** | **Catalog #227 substrate_composition_matrix default; unwind path: sister probe-disambiguator at `tools/probe_wyner_ziv_composition_alpha_disambiguator.py` is the canonical empirical arbiter pending L1 paired smoke. SATURATING (alpha=0.5) is the realistic prior per the sister primitive's design memo §Predicted ΔS band.** |
| **MLX training + numpy-portable inflate byte-identical composition** | **CARGO-CULTED** | **Sister Z6-v2 + DP1 demonstrate the pattern but per-substrate FP16-vs-FP32 + endian + alignment can drift. Unwind: L1 first-smoke MUST run encode (MLX) + inflate (numpy) on the SAME source bytes + assert byte-identity.** |
| MLX-first directive applies to all training | HARD-EARNED | CLAUDE.md 8th MLX-FIRST standing directive 2026-05-26 reinforced 2026-05-27 + sister Z6-v2 landing pattern. |
| HNeRV parity L4 ≤200 LOC + ≤2 deps inflate runtime | HARD-EARNED | CLAUDE.md HNeRV parity discipline L4 + sister D1/D4/DP1 landings. |
| `monolithic_single_file_wzpsc01` is the right archive grammar | HARD-EARNED | HNeRV parity L3 (monolithic single-file) + sister substrate landings (D1 WZF0, DP1 DP1\\x00 follow same pattern). |
| `intercept_location=STATE_DICT_SERIALIZATION` is the highest-leverage intercept | HARD-EARNED | sister primitive's empirical anchors: pr101_state_dict + pr106_state_dict prober showed 0.47 score-savings each. |

## 9-dimension success checklist evidence

Per Catalog #294 + CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" non-negotiable:

1. **UNIQUENESS** — pipeline-stage codec wrapper that composes the canonical
   primitive into a substrate per Catalog #241/#242 schema. Structurally distinct
   from sister substrates `d4_wyner_ziv_frame_0` (frame-pair surface),
   `wyner_ziv_cooperative_receiver` (bit-plane DISCUS), and the primitive itself
   (which is a black-box codec, not a substrate).

2. **BEAUTY + ELEGANCE** — 5 files / ~1270 LOC total (incl. docstrings + tests):
   `__init__.py` (canonical contract registration) + `architecture.py` (thin
   primitive wrapper) + `trainer.py` (MLX-first; smoke + L0 NotImplementedError)
   + `archive.py` (WZPSC01 grammar) + `inflate.py` (numpy-portable runtime).
   Reviewable in 30 seconds per PR101 medal-class precedent.

3. **DISTINCTNESS** — explicit `## Distinction from sister Wyner-Ziv substrates`
   in `__init__.py` docstring enumerates structural differences from D4
   (frame-pair surface) + wyner_ziv_cooperative_receiver (DISCUS bit-plane) +
   the primitive itself (substrate-scope wrapper vs codec-primitive). Distinct
   archive MAGIC `WZPSC` (vs sister `WZF0` / `DPCS` / `z6v2cu1`).

4. **RIGOR** — premise verification per Catalog #229 (read 740-LOC primitive +
   sister Z6-v2 substrate + d4 substrate + canonical SubstrateContract +
   contract field invariants before authoring); 18 dedicated tests pin contract
   schema + observability + roundtrip + Catalog #205 device + Catalog #240
   transparent non-dispatchable; adversarial Contrarian + Assumption-Adversary
   council surfaced HARD-EARNED vs CARGO-CULTED classification per assumption.

5. **OPTIMIZATION PER TECHNIQUE** — substrate-OPTIMAL engineering per layer
   (table above): the WZ-pipeline-stage codec mathematical structure (Wyner
   1976 + Atick-Redlich + Tishby) shapes every decision. NOT shared-helper
   shortcut from Z6-v2 OR Z4 cooperative-receiver. Per-substrate routing per
   `score_aware_loss="custom"` enables substrate-specific Atick-Redlich
   MI-max binding at L1.

6. **STACK-OF-STACKS COMPOSABILITY** — this substrate's canonical insertion
   point (any base substrate's pre-entropy stage) makes it ORTHOGONALLY
   composable with PR101 / PR106 / A1 / FEC6 / Cascade A / NSCS06 v8 chroma_lut
   etc. Catalog #227 substrate_composition_matrix alpha is the canonical
   composition gate; default alpha=1.0 (additive prior) refined to alpha=0.5
   (saturating) by sister probe-disambiguator + paired-smoke at L1.

7. **DETERMINISTIC REPRODUCIBILITY** — frozen `WynerZivPipelineStageCodecArchitecture`
   dataclass + frozen `WynerZivLayerConfig` + `deterministic_seed` field +
   byte-identical encode roundtrip pinned by test
   `test_encode_decode_roundtrip_byte_identical`; archive grammar deterministic
   per `encode_archive_bytes_scaffold` (no timestamps, sorted-keys JSON meta).

8. **EXTREME OPTIMIZATION + PERFORMANCE** — compress-time cost amortized
   across all dispatches; inflate-time overhead is one lzma.decompress call
   on the side stream (~ms wall-clock per smoke evidence:
   `decoder_complexity_estimate_seconds=2.3e-5`). Numpy-portable inflate +
   no MLX runtime dep keeps the runtime closure ≤2 deps per HNeRV parity L4.

9. **OPTIMAL MINIMAL CONTEST SCORE** — predicted ΔS band derived per
   §Predicted ΔS band below; L0 evidence_grade='predicted' + score_claim=False
   + promotable=False per Catalog #323 + Catalog #341 Tier A markers.
   Promotion to a contest score claim requires L1 paired CUDA+CPU auth-eval
   per Catalog #246. Per Wyner 1976 theoretical bound: R(D|Y) ≤ R(D) with
   equality iff X ⫫ Y (independence); the achievable savings depend
   empirically on Y-derivable-prefix density.

## Observability surface

Per CLAUDE.md "Max observability — non-negotiable" + Catalog #305: all 6 facets
declared as a module-level dict in `__init__.py::OBSERVABILITY_SURFACE`:

1. **Inspectable per layer** — per-pair `(offset_in_y, prefix_len)` split tuple
   + per-stage byte counts exposed via `architecture.report_stage_byte_counts()`.

2. **Decomposable per signal** — per-substrate-iteration total loss decomposed
   into (1) main-stream rate term, (2) side-stream bake-in cost (inflate.py
   LOC overhead), (3) cooperative-receiver discrepancy, (4) HNeRV parity L4
   ≤200 LOC budget headroom.

3. **Diff-able across runs** — byte-stable under deterministic seed + fixed Y;
   regression test `test_encode_decode_roundtrip_byte_identical` pins.

4. **Queryable post-hoc** — training emits `wyner_ziv_pipeline_stage_codec_
   training_observability_<utc>.jsonl` with per-iteration metrics; fcntl-locked
   APPEND-ONLY per Catalog #128/#131 sister discipline.

5. **Cite-able** — every persisted artifact carries Provenance per Catalog #323
   with `(commit_sha, lane_id, substrate_id, canonical_helper_invocation=
   'tac.codec.wyner_ziv_layer.insert_wyner_ziv_layer', hardware_substrate,
   intercept_location, side_info_source)`; `score_claim=False` + `promotable=
   False` + `axis_tag='[predicted]'` at L0 per Catalog #341 Tier A markers.

6. **Counterfactual-able** — every byte in the WZ side stream paired with a
   counterfactual probe per Catalog #105/#139/#220/#272 byte-mutation-smoke
   discipline; encode-decode roundtrip IS the canonical no-op detector.

## Predicted ΔS band

Per Catalog #296 Dykstra-feasibility + Wyner 1976 first-principles bound:

**Predicted band: ΔS ∈ [-0.0470, -0.0050] [contest-CUDA & contest-CPU paired]**

Derivation:

* **Upper bound (most pessimistic; -0.0050)**: Catalog #227 saturating
  composition (alpha=0.5) with FEC6 0.19205 frontier OR Y-derivable-prefix
  density ≤ 1% on real bytes (per Assumption-Adversary CARGO-CULTED critique).
  This is the operator-deferral case: if first-smoke measures density ≤ 1%,
  DEFER to alternative side_info_source per op-routable #4.

* **Lower bound (most optimistic; -0.0470)**: Catalog #227 additive composition
  (alpha=1.0) with sister primitive's prober anchor (pr101_state_dict
  0.47 score-savings per substrate at lzma ratio 0.217-0.228). This requires
  Y-derivable-prefix density ≥ ~20% on real bytes (matching prober anchor).

* **Dykstra-feasibility check**: the Wyner 1976 R(D|Y) achievable rate is
  the canonical first-principles bound. Convex feasibility region is the
  intersection of `R ≤ R(D|Y)` (rate constraint) ∩ `D ≤ D_target` (distortion
  constraint) ∩ `inflate.py LOC ≤ 200` (HNeRV parity L4 budget) ∩
  `runtime_dep_closure ≤ 2 deps` (HNeRV parity L4). The L0 scaffold's
  architecture lives in this feasible region by construction; the L1 trainer
  empirically measures where on the R(D|Y) curve we land.

**Validation status**: `pending_post_training` per Catalog #324. The L1
trainer's first-smoke measures Y-derivable-prefix density + lzma ratio on
real bytes; the L2 promotion requires paired CUDA+CPU auth-eval per Catalog
#246. NO promotion until empirical evidence.

## Reactivation criteria (per CLAUDE.md "Forbidden premature KILL")

Per CLAUDE.md "KILL is the LAST RESORT" + "Forbidden premature KILL without
research exhaustion": if L1 first-smoke measures Y-derivable-prefix density
that falsifies the predicted band, the substrate is DEFERRED-PENDING-
research, NOT killed. Reactivation paths (priority-ordered):

1. **L1 first-smoke with Comma2k19 Y-derivation** (canonical default; ~$0.30
   Modal T4 5min). Measures Y-derivable-prefix density on real PR101 fp16
   state_dict bytes. Predicted cost $0.30. Tests Assumption-Adversary
   CARGO-CULTED hypothesis #1 (cross-substrate Y derivation density).

2. **L1 first-smoke with torch_defaults / math_constants Y-derivation**
   (fallback if Comma2k19 density ≤ 1%). Predicted cost $0.30 each; total
   $0.90 envelope. Tests alternative canonical Y sources per the primitive's
   taxonomy.

3. **L1 first-smoke with per-pair PoseNet-output side-info via attestation**
   (Catalog #320 strict-scorer-rule attestation path). Predicted cost ~$0.50
   (requires PoseNet inflate-time forward; CUDA-only). The Atick-Tishby-Wyner
   triple's DEEPEST class-shift route per Catalog #311.

4. **Composition matrix alpha measurement via sister probe-disambiguator**
   (`tools/probe_wyner_ziv_composition_alpha_disambiguator.py`). Empirically
   resolves alpha between THIS substrate + FEC6 frontier; gates L2 promotion
   per Catalog #227 + #319 sister deliverability-proof discipline.

If all 4 reactivation paths return density ≤ 1% AND alpha ≤ 0.3 (saturating),
the substrate is ARCHIVED-pending-paradigm-shift per CLAUDE.md "Substrate
retirement discipline" + Catalog #298. The paradigm (Wyner 1976 pipeline-stage
codec) is INTACT per Catalog #307 IMPLEMENTATION-LEVEL vs PARADIGM-LEVEL
falsification — the falsification would be of THIS specific composition
mechanism, not the Wyner-Ziv paradigm.

## Catalog #324 predicted_band_validation_status

`pending_post_training` — no post-training Tier-C density measurement at L0.
The L1 trainer's first-smoke produces the post-training Tier-C anchor that
validates (or refines) the predicted band per Catalog #324
post-training-validation discipline.

## 6-hook wire-in declaration (per Catalog #125)

* hook #1 sensitivity-map = **ACTIVE** (per-pair byte-coverage exposed via
  `architecture.report_stage_byte_counts()`; routes through sister Wire-in #2
  `tac.sensitivity_map.wyner_ziv_reweight`)
* hook #2 Pareto constraint = **ACTIVE** (`inflate_runtime_loc_budget=200`
  per HNeRV parity L4 waiver; `side_bytes_compressed_baked ≤ side_info_max_bytes`
  per primitive contract)
* hook #3 bit-allocator = **ACTIVE** (main_compressed + side_compressed_baked
  reduce per-substrate byte budget via canonical
  `tac.codec.wyner_ziv_layer.insert_wyner_ziv_layer`)
* hook #4 cathedral autopilot dispatch = **ACTIVE** via auto-discovery per
  Catalog #335 canonical contract + Catalog #336 invocation discipline
* hook #5 continual-learning posterior = **ACTIVE** via
  `tac.council_continual_learning.append_council_anchor` per Catalog #355
  meta-Lagrangian wire-in
* hook #6 probe-disambiguator = **ACTIVE**; canonical sister
  `tools/probe_wyner_ziv_composition_alpha_disambiguator.py` IS the
  apples-to-apples arbiter between additive (α=1.0) vs saturating (α=0.5)
  composition with the FEC6 0.19205 frontier per Catalog #227.

## Cross-references

* `tac.codec.wyner_ziv_layer` — canonical primitive (lane
  `lane_wyner_ziv_pipeline_stage_codec_primitive_20260517` at L1; 740 LOC +
  64 tests; landed 2026-05-17). THIS substrate is its canonical substrate-scope
  consumer per Catalog #335 + Catalog #341.
* `tac.substrates.d4_wyner_ziv_frame_0` — sister substrate (frame-pair
  surface; SE(3) parametric motion + photometric residual).
* `tac.substrates.wyner_ziv_cooperative_receiver` — sister substrate (DISCUS
  bit-plane Slepian-Wolf).
* `tac.substrates.z6_v2_cargo_cult_unwind` — sister L0 SCAFFOLD landing
  pattern (Z6-v2 cargo-cult-unwind 2026-05-27 commit `afa5ba837`).
* `.omx/research/n1_path_5_8_seed_bootstrap_k5_landed_20260528T014859Z.md` —
  N1 path 5 STRUCTURAL CEILING REINFORCED memo identifying THIS substrate
  as reactivation path #3.
* `.omx/research/grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517.md` —
  T3 grand council symposium on Wyner-Ziv contest-compliance.
* `.omx/research/wyner_ziv_optimal_implementation_queue_20260517.md` —
  Q1-Q5 implementation queue from the symposium (this substrate operationalizes
  Q-substrate-scope-consumer; Q1 prover + Q2 autopilot-reweight already landed).
* CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council
  symposium" non-negotiable + Catalog #325 6-step contract.
* CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L1-L13.
* Wyner & Ziv 1976 "The Rate-Distortion Function for Source Coding with Side
  Information at the Decoder" IEEE Trans IT-22(1):1-10.
* Atick & Redlich 1990 "Towards a Theory of Early Visual Processing" Neural
  Computation 2(3):308-320.
* Tishby & Zaslavsky 2015 "Deep Learning and the Information Bottleneck
  Principle" IEEE ITW.
