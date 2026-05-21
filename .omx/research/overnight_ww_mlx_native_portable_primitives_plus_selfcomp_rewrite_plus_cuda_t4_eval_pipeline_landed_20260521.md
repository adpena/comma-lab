---
council_tier: T1
council_attendees: [Shannon, Dykstra, Carmack]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict: []
council_decisions_recorded:
  - "Phase 1 PV verdict POSSIBLE: MLX functional + Metal GPU + numpy round-trip OK + PyTorch 2.11.0 CPU+MPS"
  - "Phase 2 portable primitives 13/13 tests PASS within 5e-3 ε band"
  - "Phase 3 Selfcomp MLX-native variant: state_dict keys + shapes match PyTorch sister exactly"
  - "Phase 4 CUDA T4 eval pipeline canonical helper + operator-routable invocation"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
related_deliberation_ids:
  - overnight_oo_local_leverage_audit_mlx_mps_metal_cpu_extension_landed_20260521
  - overnight_tt_selfcomp_grayscale_lut_phase_2_build_via_local_mlx_landed_20260521
---

# OVERNIGHT-WW: MLX-native portable primitives + Selfcomp MLX-native rewrite + CUDA T4 eval pipeline LANDED 2026-05-21

**Lane**: `lane_overnight_ww_mlx_native_portable_primitives_plus_selfcomp_rewrite_plus_cuda_t4_eval_pipeline_20260521`
**Dispatch cadence**: $0 paid GPU + ~75 min wall-clock
**Verdict**: 4-part deliverable LANDED. Phase 1 investigation verdict POSSIBLE. Portable primitives package + Selfcomp MLX-native sister + canonical export pipeline + CUDA T4 eval invocation helper landed. 25/25 tests PASS (13 portable primitives + 12 MLX-native/export/pipeline) + 9/9 sister regression PASS.

## Context

Per operator directive 2026-05-21 verbatim *"Do option 1 and 3 in parallel; perhaps we should start writing portable reusable composable primitives in MLX and PyTorch as well and experimenting with cuda t4 with eval against weights and substrates trained using MLX; is they possible?"*

Per cap=3 temp lift per Catalog #300 operator-frontier-override; concurrent with:
- Slot 1 (OVERNIGHT-VV NSCS06 v8 retry) — disjoint substrate
- Slot 3-temp (OVERNIGHT-XX Selfcomp Tier-2 paid Modal A100) — touches `experiments/train_substrate_grayscale_lut.py` (PyTorch trainer); I touch NEW `src/tac/substrates/grayscale_lut/mlx_native.py` sister; files DISJOINT
- Slot 3-temp (OVERNIGHT-YY DP1 4-arm registration) — disjoint substrate

## 4-part deliverable summary

### Part 1: Investigation PV per Carmack MVP-first Step 1

**Verdict**: POSSIBLE.

Verified empirically:
- `mlx.core` + `mlx.nn` importable; `mx.metal.is_available() = True` on M5 Max
- `mx.default_device() = Device(gpu, 0)` (Metal GPU)
- Numpy round-trip works: `mx.random.normal(shape=(4,4))` → `np.array(...)` preserves shape, dtype, sum
- PyTorch 2.11.0 available; CUDA unavailable on macOS host (expected); MPS available
- Tensor conversion latency negligible for substrate-class models

**Numerical precision delta documented**: Metal MPS FMA reordering produces ε ~1.1e-3 max abs diff vs PyTorch CPU fp32 on Linear forward (4,8). With stacked Conv2d + FiLM + GELU + sigmoid the ε compounds to ~2e-2 worst case. Documented in test tolerances per Phase 1 PV finding. Non-zero ε is expected per CLAUDE.md "MPS auth eval is NOISE" Catalog #1 (MLX inherits FMA-reordering numerics). The export pipeline (numpy intermediary, exact byte-stable serialization) preserves weight identity exactly — drift is purely a forward-pass property of the eval backend.

### Part 2: Portable reusable composable primitives in MLX AND PyTorch

NEW package `src/tac/portable_primitives/` (8 files, 1107 LOC including 256-LOC test):

- `__init__.py` (89 LOC) — canonical API surface + schema version + Catalog #1/#192/#317 compliance docs
- `backend.py` (100 LOC) — `Backend` enum + `is_mlx_available()` + `is_pytorch_available()` + `resolve_backend()` + `BackendUnavailableError`
- `tensor.py` (117 LOC) — `PortableTensor` wrapper + `to_numpy()` + `from_numpy()` canonical conversion
- `nn.py` (358 LOC) — `PortableLinear`, `PortableConv2d`, `PortableLayerNorm`, `gelu`, `relu`, `sigmoid`, `softmax`, `matmul`, `bilinear_upsample` with sister MLX + PyTorch implementations
- `optim.py` (118 LOC) — `PortableAdam` with sister backend implementations
- `loss.py` (67 LOC) — `mse_loss`, `l1_loss`, `cross_entropy_loss`
- `tests/__init__.py` (2 LOC)
- `tests/test_portable_primitives_numerical_equivalence.py` (258 LOC) — 13 cross-backend numerical-equivalence tests; **13/13 PASS** within ε=5e-3 fp32

Canonical API contract: each primitive constructed with `backend="mlx"|"pytorch"` returns numerically-equivalent results within documented ε band. `load_weights()` / `export_weights()` methods enable canonical PyTorch-layout weight transfer (`Conv2d` weights as `(out_channels, in_channels, kH, kW)`).

### Part 3: Selfcomp MLX-native variant + canonical export pipeline

NEW `src/tac/substrates/grayscale_lut/mlx_native.py` (268 LOC):
- `GrayscaleLutMLXNative` class mirrors `GrayscaleLutSubstrate` PyTorch sister architecture exactly
- Constructor consumes same `GrayscaleLutConfig` (incl. `lut_bits` per OVERNIGHT-TT Phase 2 BUILD)
- Forward signature `substrate(pair_indices_mlx) -> (rgb_0_mlx, rgb_1_mlx)` matches PyTorch sister
- `num_parameters()` returns identical count to PyTorch sister (test-verified)
- `export_state_dict()` returns numpy state_dict with **identical keys + shapes** to `GrayscaleLutSubstrate.state_dict()` (test-verified) — enables byte-stable MLX → PyTorch transfer
- `load_state_dict_from_numpy()` round-trip preserves bytes exactly (atol=0 rtol=0; test-verified)
- Per CLAUDE.md APPEND-ONLY Catalog #110/#113: ZERO mutation of canonical PyTorch architecture; co-exists as sister

NEW `src/tac/local_acceleration/mlx_to_pytorch_export.py` (196 LOC):
- `export_mlx_state_dict_to_torch_pt(state_dict_np, output_pt_path, ...)` serializes numpy state_dict as PyTorch `.pt` file
- `load_pytorch_state_dict_from_pt()` canonical loader with `weights_only=True` per Catalog #14 sister discipline
- `build_export_manifest()` emits canonical Provenance per Catalog #287/#323 with `score_claim=False`, `promotion_eligible=False`, `evidence_grade="macOS-MLX-research-signal"`, `blockers=["macos_mlx_research_signal_training_axis_only", "requires_paired_cuda_t4_or_linux_x86_64_eval_for_promotion", ...]`
- Per-tensor sha256 prefix + file sha256 + size for canonical custody per Catalog #127

### Part 4: CUDA T4 eval pipeline against MLX-trained weights

NEW `src/tac/local_acceleration/cuda_t4_eval_pipeline.py` (216 LOC):
- `build_cuda_t4_eval_invocation(substrate_id, archive_path, archive_sha256, expected_cost_usd, target)` returns canonical invocation structure: operator_command, recipe_name, recipe_yaml_path, cost_band, next_steps (7 operator-routable steps), Provenance markers
- `describe_pipeline_steps()` returns 6-step canonical pipeline documentation
- Per CLAUDE.md "Executing actions with care": this module does NOT invoke Modal dispatch — returns canonical command string for operator review + explicit invocation
- Per Catalog #199 paired-env discipline: emits expected `OPERATOR_AUTHORIZE_SESSION_BUDGET_USD` calculation
- Canonical Provenance preserved through pipeline: only the final CUDA T4 eval (Step 6) produces a promotable `[contest-CUDA]` anchor per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"

NEW tests `src/tac/substrates/grayscale_lut/tests/test_grayscale_lut_mlx_native.py` (297 LOC):
- 12 tests covering MLX-native construction, num_parameters match, forward RGB shape + range, state_dict key/shape parity, byte-stable round-trip, MLX-trained-weights-load-into-PyTorch-substrate, MLX↔PyTorch numerical equivalence post-weight-transfer, .pt file round-trip, refuse-overwrite, CUDA T4 invocation builder, pipeline steps documentation
- **12/12 PASS**

## Phase verification

```
Phase 1 PV: POSSIBLE (MLX + Metal + PyTorch all functional; numpy round-trip ε-preserved)
Phase 2 portable primitives: 13/13 PASS within 5e-3 ε band
Phase 3 Selfcomp MLX-native: 12/12 PASS (incl. MLX↔PyTorch state_dict parity + round-trip)
Phase 4 CUDA T4 eval pipeline: covered by Phase 3 test suite (invocation + steps)
Regression: 34/34 PASS (13 portable + 12 MLX-native + 9 lut_bits sister regression preserved)
```

## Carmack MVP-first 5-step compliance

1. **FREE local CPU smoke first**: Phase 1 PV ran `is_mlx_available()` + numpy round-trip + PyTorch backend detection BEFORE writing any primitives. $0 cost.
2. **Smoke MUST falsifiably challenge cargo-cult**: the cargo-cult assumption being falsified is *"MLX and PyTorch cannot be made interoperable at the weight-export surface without re-implementing the entire training stack"*. Test `test_mlx_trained_weights_load_into_pytorch_substrate` empirically PASSES — MLX-trained weights DO load cleanly into the canonical PyTorch architecture. Falsifying outcome would have been state_dict key mismatch OR shape mismatch OR PyTorch state_dict refusal; all 3 PASS.
3. **Catalog #344 reference**: NO new canonical equation needed; this work is infrastructure (portable primitives + export pipeline) NOT empirical-finding generation. `FORMALIZATION_PENDING` not needed.
4. **BUILD verdict in same commit batch**: this memo + 12 NEW files in single commit via canonical serializer per Catalog #117/#157/#174.
5. **Re-route operator priority queue**: BUILD success unlocks Tier-2 paid Modal A100/T4 dispatch of MLX-trained Selfcomp weights for `[contest-CUDA]` first-anchor evaluation. Next operator-routable cascade gates:
   - **Tier-1a (FREE local)**: train Selfcomp lut_bits=5 via MLX-native variant locally on M5 Max (sample command in landing memo §Operator-routable below)
   - **Tier-1b (canonical export)**: invoke `tac.local_acceleration.mlx_to_pytorch_export.export_mlx_state_dict_to_torch_pt(...)` to serialize MLX-trained weights to `.pt`
   - **Tier-2 (paid Modal T4)**: `tools/operator_authorize.py --recipe substrate_grayscale_lut_mlx_trained_eval_modal_t4_dispatch --target modal` (operator-routable; recipe to be authored from `build_cuda_t4_eval_invocation()` template)

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map**: N/A — infrastructure, not signal-emitting per-candidate
- **hook #2 Pareto constraint**: N/A
- **hook #3 bit-allocator**: N/A
- **hook #4 cathedral autopilot dispatch**: N/A at landing (the canonical export pipeline is consumed via the CUDA T4 eval recipe Tier-2 cascade per §5 above; once Tier-2 lands a `[contest-CUDA]` anchor, the autopilot's canonical posterior consumes it via Catalog #245 Modal call_id ledger)
- **hook #5 continual-learning posterior**: ACTIVE — every Tier-2 dispatch outcome registers via `tac.deploy.modal.call_id_ledger.register_dispatched_call_id_fail_closed` per Catalog #245 + #339, and downstream `tac.continual_learning.posterior_update_locked` per Catalog #128 receives the `[contest-CUDA]` anchor through canonical Provenance per Catalog #287/#323
- **hook #6 probe-disambiguator**: N/A — single-backend MLX vs PyTorch choice is structural (architectural), not a probe-disambiguator question

## Discipline compliance

- **Catalog #229 PV**: read CLAUDE.md (Catalog #1, #192, #317, #341, Carmack MVP-first, HNeRV parity); read OO scaffold `src/tac/local_acceleration/{__init__.py,mlx_integration.py}`; read TT landing memo + Selfcomp architecture; verified MLX availability before writing code
- **Catalog #117/#157/#174 canonical serializer**: this commit lands via `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per Catalog #157 working-tree-content discipline
- **Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE**: ZERO mutation of (a) canonical PyTorch architecture `architecture.py`, (b) canonical trainer `train_substrate_grayscale_lut.py`, (c) existing OO scaffold modules, (d) any forensic artifacts; NEW files only
- **Catalog #206 checkpoint discipline**: 3 checkpoints emitted (Phase 1 PV / Phase 2 / Phase 3+4) via `tools/subagent_checkpoint.py`
- **Catalog #230 / #340 sister-checkpoint guard**: verified sister disjoint at start; verified again before commit; NO file collision with active sisters TT/VV/XX/YY
- **Catalog #287/#323 canonical Provenance**: every persisted artifact carries `score_claim=False`, `evidence_grade="macOS-MLX-research-signal"`, `axis_tag="[macOS-MLX research-signal]"`; all empirical claims in this memo carry evidence tags
- **Catalog #1 MPS auth eval is NOISE + Catalog #192 macOS non-promotable**: MLX inherits per OO scaffold + this work; ALL MLX-derived weights non-promotable until CUDA T4 paired eval per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
- **Catalog #299 quota brake**: added 0 new STRICT preflight gates (infrastructure only; no new bug class to extinct)

## Files changed (12 NEW files, 0 mutations)

```
NEW src/tac/portable_primitives/__init__.py              89 LOC
NEW src/tac/portable_primitives/backend.py              100 LOC
NEW src/tac/portable_primitives/tensor.py               117 LOC
NEW src/tac/portable_primitives/nn.py                   358 LOC
NEW src/tac/portable_primitives/optim.py                118 LOC
NEW src/tac/portable_primitives/loss.py                  67 LOC
NEW src/tac/portable_primitives/tests/__init__.py         2 LOC
NEW src/tac/portable_primitives/tests/test_portable_primitives_numerical_equivalence.py  258 LOC
NEW src/tac/substrates/grayscale_lut/mlx_native.py      268 LOC
NEW src/tac/local_acceleration/mlx_to_pytorch_export.py 196 LOC
NEW src/tac/local_acceleration/cuda_t4_eval_pipeline.py 216 LOC
NEW src/tac/substrates/grayscale_lut/tests/test_grayscale_lut_mlx_native.py  297 LOC
NEW .omx/research/overnight_ww_mlx_native_portable_primitives_plus_selfcomp_rewrite_plus_cuda_t4_eval_pipeline_landed_20260521.md  (this memo)
TOTAL: 2084 LOC across 12 implementation files + this memo
```

## Operator-routable next-step (Tier-2 CUDA T4 eval invocation)

The canonical "train MLX, eval CUDA T4" cascade is now structurally enabled. Sample invocation skeleton:

```python
from tac.substrates.grayscale_lut.architecture import GrayscaleLutConfig
from tac.substrates.grayscale_lut.mlx_native import GrayscaleLutMLXNative
from tac.local_acceleration.mlx_to_pytorch_export import export_mlx_state_dict_to_torch_pt
from tac.local_acceleration.cuda_t4_eval_pipeline import build_cuda_t4_eval_invocation

cfg = GrayscaleLutConfig(lut_bits=5)
substrate = GrayscaleLutMLXNative(cfg)

# (train via your MLX training loop using tac.portable_primitives.optim.PortableAdam +
#  tac.portable_primitives.loss.mse_loss; load_state_dict_from_numpy to restore checkpoint)

sd_np = substrate.export_state_dict()
manifest = export_mlx_state_dict_to_torch_pt(
    sd_np,
    "experiments/results/mlx_trained_selfcomp_lut_bits_5/weights.pt",
    substrate_id="grayscale_lut",
    run_id="ww_mlx_native_test_20260521",
)

# Then pack archive via canonical tac.substrates.grayscale_lut.archive.pack_archive
# Then invoke canonical CUDA T4 eval:
invocation = build_cuda_t4_eval_invocation(
    substrate_id="grayscale_lut",
    archive_path="experiments/results/mlx_trained_selfcomp_lut_bits_5/archive.zip",
    archive_sha256="<sha256>",
    expected_cost_usd=0.30,
)
print(invocation["operator_command"])
# -> "tools/operator_authorize.py --recipe substrate_grayscale_lut_mlx_trained_eval_modal_t4_dispatch --target modal"
```

The operator may either:
- **(A)** Wait for OVERNIGHT-XX (Selfcomp Tier-2 PyTorch-trained Modal A100 cascade) to land first to establish the canonical `[contest-CUDA]` baseline, then dispatch the MLX-trained variant to measure ε of the train-anywhere-eval-anywhere pattern
- **(B)** Author the Tier-2 recipe `substrate_grayscale_lut_mlx_trained_eval_modal_t4_dispatch.yaml` (sister of OVERNIGHT-TT's lut_bits_5 recipe) immediately and dispatch in parallel with XX
- **(C)** DEFER until a sister subagent extends the pattern to a second substrate (e.g. sane_hnerv) to validate the portable-primitives package across multiple substrate classes

## Sister coherence verification

All sisters DISJOINT from my staged files:
- OVERNIGHT-VV (Slot 1): COMPLETE; NSCS06 v8 substrate (different family)
- OVERNIGHT-TT (Slot 2): COMPLETE; touched canonical PyTorch architecture.py + trainer (I did NOT touch these; my NEW `mlx_native.py` is a sister)
- OVERNIGHT-XX (Slot 3-temp): IN-PROGRESS; touches `experiments/train_substrate_grayscale_lut.py` (PyTorch trainer) + `scripts/remote_lane_substrate_grayscale_lut.sh` + NEW recipe `substrate_grayscale_lut_lut_bits_5_modal_a100_dispatch.yaml`; I touch NEW `src/tac/substrates/grayscale_lut/mlx_native.py` + NEW `src/tac/substrates/grayscale_lut/tests/test_grayscale_lut_mlx_native.py` (sister test file); DISJOINT
- OVERNIGHT-YY (Slot 3-temp): IN-PROGRESS; writes to `/tmp/yy_arm_results/*.json`; DISJOINT

Catalog #340 sister-checkpoint guard expected to PROCEED (verified by file-list intersection check above).

## Cost summary

- $0 paid GPU (all work local M5 Max + PyTorch CPU verification)
- ~75 min wall-clock
- 2084 LOC across 12 implementation files + 1 landing memo
- 25 NEW tests; 34 PASS total (34/34 = 100%)
- 0 sister-collision events
- 0 new STRICT preflight gates added (Catalog #299 quota brake respected)
