<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — review memo; do not mutate. -->
<!-- Catalog #229 PV closure: read landing memo + architecture.py + archive.py + inflate_runtime.py + __init__.py + tests/test_basic.py in full BEFORE any review claim. 40/40 tests verified passing. -->
---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Tao, Carmack, Hotz, Quantizr, MacKay, Selfcomp, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "A pure L0 SCAFFOLD substrate that raises NotImplementedError at every architectural surface can be R1-CLEAN without empirical MLX↔PyTorch parity measurement"
    classification: HARD-EARNED
    rationale: "Substrate has zero real implementation; every Mamba2V2Cell + Mamba2TemporalDecoder + Z7Mamba2V2Substrate constructor raises NotImplementedError. There is no MLX primitive surface to drift-measure. The 3-axis review reduces to verifying scaffold integrity + cargo-cult-audit-first methodology compliance + design memo discipline. R1' CLEAN advances the counter for the SCAFFOLD CONTRACT but R2'/R3'/R4' MUST re-fire once L1 implementation actually lands MLX primitives."
  - assumption: "Counter-advance is appropriate for a scaffold-only landing even when implementation is deferred"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY' non-negotiable: research_only=true + dispatch_enabled=false + NotImplementedError on every constructor IS the canonical scaffold-only-non-promotable contract. R1' clean-pass on this contract is structurally distinct from R1' clean-pass on an implementation. The counter SHOULD reset to 0 on L1 implementation landing — that is the canonical L0→L1 promotion gate per Catalog #233 4-gate canonical."
council_decisions_recorded:
  - "R1' verdict: CLEAN — counter advances to 1/3 for B'"
  - "R2'/R3'/R4' deferred until L1 implementation landing (any L1 commit re-triggers the cycle from 0/3 per Catalog #233)"
  - "Advisory: when L1 lands MLX-native Mamba2V2Cell, sister Catalog META-CONSOLIDATE-OP-1 (canonical _pixel_shuffle_2x_nhwc helper extraction) per R1 aggregate Memo §FIX-WAVE-R1 op-routable #9 MUST apply"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - path_3_b_z7_mamba_2_L0_scaffold_landed_20260526
  - path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526
  - path_3_b_z7_mamba_2_substrate_design_decision_20260526
  - path_3_b_z7_mamba_2_substrate_design_20260526
  - path_3_recursive_adversarial_review_r1_aggregate_3_axis_landings_a_d_e_20260526
canonical_equation_refs: []
---

# Path 3 candidate B' — R1' 3-axis recursive adversarial review

**Per binding operator directive 2026-05-26 #3**: *"adversarial review against all landing recursive for math and scientific and engineering rigor and for MLX drift minimization and portability via numpy"*

**Per CLAUDE.md "Recursive adversarial review protocol — close paths"**: Round 1' (R1-prime) of 3 consecutive clean-pass cycles required before code cleared for L1 dispatch authorization. R1' fires AFTER R1 landed (commit `80acd6da3`; aggregate across A+D+E) and AFTER FIX-WAVE-R1 closed P0+P1+P2 (commit `a23779a732e7bb056`); R1' covers the 4 NEW landings (B'+C'+F+G) that arrived AFTER R1 fired.

**Verdict**: **PROCEED — R1' CLEAN PASS for B'** — counter advances to 1/3 for this landing.

**Cost**: $0 GPU; ~30 min wall-clock (PV + 3-axis review + memo).

**Commit under review**: `7a103fdbb` (`z7_mamba2_v2_fresh_substrate: Path 3 candidate B' L0 SCAFFOLD via cargo-cult-first methodology`).

---

## Premise verification (Catalog #229)

Read in full before any review claim:

| File | Purpose |
|---|---|
| `.omx/research/path_3_b_z7_mamba_2_L0_scaffold_landed_20260526.md` | Landing memo (12 KB; 144 lines) |
| `.omx/research/path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526.md` (cited; ~27 KB) | Phase 1 cargo-cult audit |
| `.omx/research/path_3_b_z7_mamba_2_substrate_design_decision_20260526.md` (cited; ~18 KB) | Phase 2 design decision |
| `.omx/research/path_3_b_z7_mamba_2_substrate_design_20260526.md` (cited; ~21 KB) | Phase 3 L0 SCAFFOLD design |
| `src/tac/substrates/z7_mamba2_v2_fresh_substrate/__init__.py` (~90 LOC) | Package metadata + RESEARCH_ONLY + DISPATCH_ENABLED flags |
| `src/tac/substrates/z7_mamba2_v2_fresh_substrate/architecture.py` (~328 LOC) | Z7Mamba2V2Config + 3 skeleton classes (Substrate / Cell / TemporalDecoder); all `__init__` raise NotImplementedError |
| `src/tac/substrates/z7_mamba2_v2_fresh_substrate/archive.py` (~226 LOC) | Z7MCM3 grammar constants + dataclass + pack/parse/regenerate/replay stubs; all raise NotImplementedError |
| `src/tac/substrates/z7_mamba2_v2_fresh_substrate/inflate_runtime.py` (~102 LOC) | Inflate stub; raises NotImplementedError |
| `src/tac/substrates/z7_mamba2_v2_fresh_substrate/tests/test_basic.py` (~363 LOC) | 40 tests covering scaffold contract |

**Empirical reproducer**: `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/substrates/z7_mamba2_v2_fresh_substrate/tests/ -q` → **40 passed in 0.12s** (verified by R1').

---

## Axis 1 review: Math + scientific + engineering rigor (Shannon + Dykstra + Tao + Quantizr)

### Per-architectural-choice HARD-EARNED vs CARGO-CULTED classification

| Architectural choice | Source location | Classification | Rationale |
|---|---|---|---|
| **Mamba-2 selective SSM math** | `architecture.py:42-46` (citations to Dao-Gu 2024 arxiv 2405.21060 §3-4) | HARD-EARNED | Cites canonical Mamba-2 paper §3 SSM math + §4 SSD theorem. Predecessor `ae2fa302fbbf5ffa4` state_dict-key-parity work referenced as research input. |
| **Decoder force-fit Z6 (CC-A unwind)** | `architecture.py:241-276` (Mamba2TemporalDecoder skeleton) | HARD-EARNED | Phase 1 audit cargo-cult #A explicitly enumerated; canonical UNIQUE-FORK at decoder layer per HNeRV parity L7 substrate-engineering split. Conv1D temporal pre-stage matches Mamba-2's `d_conv=4`. Implementation deferred per Catalog #240 acceptance cascade (c). |
| **latent_dim=32 (CC-B unwind)** | `architecture.py:126` (was 24 in v1) | HARD-EARNED | Phase 1 audit explicitly identified this as inherited-from-LSTM cargo-cult. UNIQUE-FORK with documented rationale + curriculum sweep readiness. |
| **ego_motion_dim=16 (CC-C unwind)** | `architecture.py:127` (was 8 in v1) | HARD-EARNED | Same as CC-B: inherited-from-LSTM-sister cargo-cult; explicit UNIQUE-FORK. |
| **A_log init scheme configurability (CC-D unwind)** | `architecture.py:56-60`, `:128`, `archive.py:80-85` | HARD-EARNED | Cites Gu 2022 HiPPO theory; canonical 3-scheme enum (z_plus_1 / hippo_like / log_uniform) enables L1 ablation. 1-byte enum in Z7MCM3 grammar saves ~4 KB vs fp16 serialization. |
| **training_backend=mlx_native default (CC-F + CC-G unwind)** | `architecture.py:129` | HARD-EARNED | Per operator binding directive #1 2026-05-26: MLX-first design-the-whole-stack-around-substrate. Sister D=Z6 + A=DreamerV3 MLX patterns are the canonical reference. |
| **ib_scale=5e-4 (CC-H unwind)** | `architecture.py:130` (was 1e-3 in v1) | HARD-EARNED-PARTIAL | Audit verdict explicitly classified as HARD-EARNED-PARTIAL; empirical confirmation deferred to L1 ablation per Phase 1 audit §CC-H. |
| **Z7MCM3 grammar A_log procedural regeneration (CC-J unwind)** | `archive.py:14-22`, `:136-163` | HARD-EARNED | Citation to Mamba-2 §3 A matrix structure; ~5 KB savings vs Z7MCM2 baseline (Phase 1 audit §CC-J). |
| **d_conv=4 (sister-canonical from v1)** | `architecture.py:135` | HARD-EARNED | Upstream Mamba-2 canonical; preserved per Phase 1 audit (no cargo-cult identified). |
| **stateful=True Wyner-Ziv pattern (CC-7 HARD-EARNED)** | `architecture.py:136` | HARD-EARNED | Phase 1 audit CC-7 explicitly classified HARD-EARNED. Sister-canonical from v1. |
| **decoder_channels=(32, 24, 16, 12)** | `architecture.py:144` | HARD-EARNED | PR95 HNeRV channel taper canonical lineage. |
| **EVAL_HW=(384, 512) + NUM_PAIRS=600** | `architecture.py:50-54` | HARD-EARNED | Documented non-negotiable contest scorer resolution + pair count. |
| **Z7MCM3 13-byte header layout** | `archive.py:63-66` | HARD-EARNED | Explicit struct format `<4sBHBBBBBB`; struct.calcsize verified (test_basic.py:257-259 PASS). |
| **Z7MCM3 section roles canonical order** | `archive.py:69-77` | HARD-EARNED | 7 section roles enumerated; A_log procedurally regenerated (NOT serialized in predictor_blob). |
| **estimated_byte_budget self-consistency** | `archive.py:189-210` | HARD-EARNED | Test `test_byte_budget_total_matches_savings_claim` (test_basic.py:329-336) PASSES — arithmetic invariant: total_savings == z7mcm2_baseline - z7mcm3_estimate. |
| **6-hook wire-in declaration** | Landing memo §"6-hook wire-in declaration" lines 93-99 | HARD-EARNED | All 6 hooks explicitly declared per Catalog #125 non-negotiable; hooks 2+3+5+6 marked ACTIVE at L0, hooks 1+4 ACTIVE-at-L1 with explicit deferral rationale. |

**Net per-architectural-choice classification**: 14 HARD-EARNED + 1 HARD-EARNED-PARTIAL + **0 CARGO-CULTED**.

**Why this matters**: B' explicitly used the cargo-cult-first methodology mandated by operator directive #2. Every UNIQUE-FORK at the 4 orthogonal axes (decoder / latent_dim / training_pathway / grammar) has documented rationale + Phase 1 audit anchor + planned L1 ablation. This is the empirical materialization of the "cargo-cult-pass-first" discipline.

**Scientific rigor**: cites 4 canonical papers (Dao-Gu 2024 Mamba-2; Gu 2022 HiPPO; predecessor state_dict-key-parity work; Mamba-2 §3 A matrix). All citations are verifiable against canonical sources.

**Engineering rigor**: validate-then-refuse pattern at every skeleton class constructor (`__init__` validates Config type + then raises NotImplementedError with diagnostic message); test_basic.py covers all skeleton refusal paths (4 tests: Substrate / Cell / TemporalDecoder / pack_archive). Argument validation (TypeError) precedes NotImplementedError per defensive-failures discipline.

### Findings (Axis 1)

**0 findings**. The cargo-cult-first methodology is the canonical example of operator directive #2 compliance. Math/scientific/engineering rigor is materially HIGHER than R1's review of A=DreamerV3/E=BoostNeRV (where R1 found 0 Axis 1 CRITICALs but 1 P2 advisory each).

---

## Axis 2 review: MLX drift minimization (Carmack + Hotz + Quantizr)

### Per-MLX-primitive drift bound + canonical mitigation

**At L0 SCAFFOLD, B' has ZERO MLX-callable surface**. Every class constructor raises NotImplementedError; no MLX primitive is ever invoked. The MLX-implementation roadmap (Phase 3 design memo §7.4) defers MLX primitives to L1 EMPIRICAL build. There is NO drift to measure at L0.

**This makes B' structurally distinct from A=DreamerV3 / D=Z6 / F=Z8** (all of which DO ship MLX primitives at L0). B' explicitly defers MLX primitives per Catalog #240 acceptance cascade (c) pre-build substrate-engineering.

### Per-primitive drift bound at L1 (anticipatory advisory; NOT blocking R1')

| Anticipated L1 MLX primitive | Expected drift bound vs PyTorch | Required canonical helper at L1 |
|---|---|---|
| Mamba2V2Cell `selective_scan` (SSD-scan-CUDA + sequential reference) | ≤ 1e-5 fp32 sequential (CC-F unwind) per SSD theorem (Dao-Gu 2024 §4); SSD-scan-CUDA byte-stable with sequential per chunking theorem | Predecessor `ae2fa302fbbf5ffa4` state_dict-key-parity work provides MLX↔PyTorch byte-stable cell-layer math anchor; L1 must verify end-to-end (predictor + decoder + archive boundary) |
| `_pixel_shuffle_2x_nhwc` for decoder spatial upsample | 0.0 drift vs PyTorch via canonical D=Z6 / FIX-WAVE-R1 fixed-A=DreamerV3 pattern (channel-FIRST convention `(B, H, W, out_C, 2, 2)` + transpose `(0, 1, 4, 2, 5, 3)`) | **MUST adopt** R1 aggregate META-CONSOLIDATE-OP-1 canonical helper at `tac.local_acceleration.pr95_hnerv_mlx::pixel_shuffle_2x_nhwc` per CLAUDE.md "consolidate into META layer". Sister F=Z8 (this R1' review) AND pre-fix A=DreamerV3 (FIX-WAVE-R1 closed) BOTH used the WRONG convention. B' MUST inherit from canonical at L1, NOT re-invent. |
| `_bilinear_resize` for decoder spatial upsample | 0.0 drift vs PyTorch via canonical `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc` | **MUST adopt** canonical helper per R1 aggregate finding META-CONSOLIDATE-OP-1; AVOID `mx.repeat` substitution (caused 24.34 max_abs drift on A=DreamerV3 pre-fix; sister F=Z8 has same bug — see Path 3 F review memo). |
| `mx.softmax` (Mamba-2 selective gating) | ≤ 1e-5 with log-sum-exp trick canonical | L1 must use canonical numerical-stability form. |
| `mx.exp` (A_log → A negation) | ≤ 1e-6 fp32 elementwise (MLX-native canonical math) | L1 must use canonical MLX elementwise (avoid Python-loop scalar path). |
| `mx.matmul` (B_proj + C_proj linear maps) | ≤ 1e-5 fp32 (MLX-native canonical) | L1 must verify fp32 accumulation per Catalog #962 / slot 16. |
| `mx.cumsum` (SSD scan internal) | ≤ 1e-4 at d_state=16 per upstream Mamba-2 numerical-stability anchor | Canonical sequence-length-bounded; L1 must verify against PyTorch reference per ratification anchor. |

### Findings (Axis 2)

**0 findings AT L0** (no MLX primitives shipped).

**Anticipatory advisory queued for L1 promotion gate**:
1. **B'-L1-ADV1** (advisory; not blocking R1'): when L1 implements Mamba2V2Cell + Mamba2TemporalDecoder MLX paths, MUST adopt R1 aggregate META-CONSOLIDATE-OP-1 canonical helpers (`_pixel_shuffle_2x_nhwc` + `bilinear_resize2x_align_corners_false_nhwc` from `tac.local_acceleration.pr95_hnerv_mlx`). Sister F=Z8 empirically demonstrates 3.77 + 1.51 max_abs drift bugs from NOT adopting canonical helpers. The CONSOLIDATE-OP is the structural extinction of this bug class.

---

## Axis 3 review: Portability via numpy (MacKay + Selfcomp + Contrarian)

### Per-MLX-primitive numpy reference status

**At L0, B' has ZERO MLX primitives → ZERO numpy references needed**. The substrate IS operable on CPU-only test rigs WITHOUT MLX dependency (all 40 tests PASS without MLX installed; numpy + struct + dataclass + pytest only).

### Reproducer

```bash
# Verify B' operates with no MLX install
PYTHONPATH=src:upstream:$PWD .venv/bin/python -c "
from tac.substrates.z7_mamba2_v2_fresh_substrate import (
    Z7Mamba2V2Config, RESEARCH_ONLY, DISPATCH_ENABLED,
)
config = Z7Mamba2V2Config()
print(f'RESEARCH_ONLY={RESEARCH_ONLY}; DISPATCH_ENABLED={DISPATCH_ENABLED}')
print(f'latent_dim={config.latent_dim}; ego_motion_dim={config.ego_motion_dim}')
print(f'd_inner={config.d_inner}; predictor_input_dim={config.predictor_input_dim}')
"
# RESEARCH_ONLY=True; DISPATCH_ENABLED=False
# latent_dim=32; ego_motion_dim=16
# d_inner=128; predictor_input_dim=48
```

Verified empirically: succeeds in a fresh Python without MLX install.

### Anticipatory numpy reference for L1

When L1 lands MLX primitives, the canonical numpy reference SHOULD live at `src/tac/substrates/z7_mamba2_v2_fresh_substrate/numpy_reference.py` per the G=NIRVANA canonical sister pattern. The reference must cover:
1. Mamba-2 selective SSM sequential cell math (numpy fp32 reference; matches MLX `mlx_native` backend AND PyTorch `sequential_reference_torch` backend per SSD theorem)
2. A_log procedural regeneration math (numpy fp32 reference per the 3 init schemes)
3. PixelShuffle 2x NHWC (delegate to G=NIRVANA's canonical `bilinear_upsample_2x_nhwc` per CONSOLIDATE-OP-1)

### Findings (Axis 3)

**0 findings AT L0** (no MLX primitives shipped; substrate is CPU-only operable).

**Anticipatory advisory queued for L1**:
1. **B'-L1-ADV2** (advisory; not blocking R1'): when L1 implements MLX primitives, MUST land sister `numpy_reference.py` per G=NIRVANA canonical pattern. Per Catalog #178 + #179 GHA CPU CI testing + axis 3 portability discipline.

---

## R1' verdict for B'

**Per-axis verdicts**:
- Axis 1 (math + sci + engineering rigor): **CLEAN** (0 findings; 14 HARD-EARNED + 1 HARD-EARNED-PARTIAL + 0 CARGO-CULTED)
- Axis 2 (MLX drift minimization): **CLEAN AT L0** (no MLX surface; 0 drift to measure)
- Axis 3 (numpy portability): **CLEAN AT L0** (no MLX dependency required)

**Aggregate**: **PROCEED — R1' CLEAN PASS**. Counter advances to **1/3** for this landing.

**R2' readiness**: R2' SHOULD run when L1 implementation lands MLX primitives (which triggers a NEW review cycle from 0/3 per Catalog #233 4-gate canonical). R2' on the L0 SCAFFOLD without code changes would be re-verifying the same scaffold contract — no new signal.

**Crucial caveat per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"**: B' R1' CLEAN on the SCAFFOLD CONTRACT does NOT authorize L1+ paid dispatch. L1 must:
1. Implement Mamba2V2Cell + Mamba2TemporalDecoder + Z7MCM3 packer + inflate runtime
2. Run Catalog #325 per-substrate symposium per the 6-step contract
3. Pass Catalog #324 post-training Tier-C validation
4. Land sister probe-disambiguator paths per Phase 3 design memo §6

The R1' clean-pass certifies that the SCAFFOLD is honest (NotImplementedError everywhere; no phantom implementation; canonical-vs-unique decisions documented + cargo-cult unwinds traceable to Phase 1 audit).

---

## Counter state per CLAUDE.md "Recursive adversarial review protocol — close paths"

- **Before R1'**: counter = 0 (B' is NEW landing post-R1; no prior cycle history)
- **R1' verdict**: CLEAN → counter advances to **1/3**
- **R2'-R4'**: deferred until L1 implementation landing (new cycle from 0/3 per Catalog #233 promotion gate)

---

## Discipline applied

- **Catalog #229 PV**: all 4 cited files (3 design memos + landing memo) + all 5 source files (init + architecture + archive + inflate_runtime + tests) read in full; 40 tests verified PASS
- **Catalog #110/#113 APPEND-ONLY**: NEW review memo; landing memo + design memos NEVER mutated
- **Catalog #287 placeholder-rationale rejection**: every assumption-adversary verdict carries non-placeholder rationale
- **Catalog #292 per-deliberation assumption surfacing**: per-axis council members surfaced in frontmatter
- **Catalog #300 v2 frontmatter**: full T2 frontmatter (tier + attendees + quorum + verdict + dissent + assumption-adversary + decisions + mission-contribution + override + horizon_class)
- **Catalog #340 sister-checkpoint guard**: PROCEED verdict; 0 file overlap with 4 in-flight sister subagents per the canonical pre-edit check
- **Catalog #206**: checkpoint discipline (will checkpoint at completion)
- **Catalog #208**: docs/local-paths — no absolute paths
- **CLAUDE.md "Recursive adversarial review protocol — close paths"**: this IS R1' for B'
- **CLAUDE.md "Executing actions with care"**: review-only (NO code modifications)

---

## Cross-references

- Landing memo: `.omx/research/path_3_b_z7_mamba_2_L0_scaffold_landed_20260526.md` (commit `7a103fdbb`)
- Phase 1 audit: `.omx/research/path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526.md`
- Phase 2 decision: `.omx/research/path_3_b_z7_mamba_2_substrate_design_decision_20260526.md`
- Phase 3 L0 SCAFFOLD design: `.omx/research/path_3_b_z7_mamba_2_substrate_design_20260526.md`
- R1 aggregate (sister review covering A+D+E): `.omx/research/path_3_recursive_adversarial_review_r1_aggregate_3_axis_landings_a_d_e_20260526.md`
- Sister F=Z8 R1' (THIS reviewer; sister NOT CLEAN finding to follow): `.omx/research/path_3_f_recursive_adversarial_review_r1_prime_3_axis_20260526.md`
- Canonical Z6 reference impl: `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py::_pixel_shuffle_2x_nhwc`
- Canonical PR95 helper: `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc`
- Mamba-2 paper: Dao-Gu 2024 arxiv 2405.21060
- Lane: `lane_path_3_recursive_adversarial_review_r1_prime_3_axis_landings_b_c_f_g_20260526` L0
