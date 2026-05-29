<!-- SPDX-License-Identifier: MIT -->

# Tinygrad-portable inflate primitive bridge — design memo

**Date:** 2026-05-29T05:49:42Z
**Lane:** `lane_slot_j_cascade_item_6_tinygrad_portable_inflate_primitive_bridge_20260529` (L0 → L1 this commit batch)
**Cascade source:** Item 6 of 7-item operator-bound cascade per `feedback_cathedral_autopilot_is_the_canonical_meta_orchestrator_proceed_with_all_7_cascade_20260528.md`
**Author:** Claude Opus 4.7 (1M context)
**Mission contribution:** `apparatus_maintenance` (closes the 3rd portability surface gap per the 8th MLX-FIRST NUMPY-PORTABLE standing directive; unblocks future substrate trainers from per-substrate-rediscovery of the canonical tinygrad bridge contract)

## Operator directive (canonical source)

Operator NON-NEGOTIABLE 2026-05-28: cascade item 6 = "TINYGRAD-PORTABLE INFLATE PRIMITIVE BRIDGE — 3rd sister surface per MLX-FIRST 8th standing directive". The 8th standing directive (per `feedback_mlx_first_numpy_portable_individually_fractally_optimized_standing_directive_20260526.md`): every substrate is MLX-first at training time AND numpy-portable at inflate time AND individually-fractally-optimized. The bridge contract: `<framework> state_dict → npz → ZIP-member at fixed offset → numpy inflate primitives`. **Inflate.py L4 ≤200 LOC budget gets 3 framework choices not 2** (MLX + PyTorch + tinygrad → numpy).

## Scope (NEW canonical surface)

`src/tac/local_acceleration/tinygrad_bridge.py` — canonical bridge module sister of `pr95_hnerv_mlx.py` (PR95 HNeRV MLX bridge) at the **tinygrad framework alternative surface**. Provides:

1. **Framework availability detection** — deferred-import `is_tinygrad_available()` + canonical fail-closed error per `Backend.TINYGRAD` selection cascade.
2. **Bridge contract operationalization** — `tinygrad_state_dict_to_zip_member_bytes(state_dict, *, archive_name)` thin wrapper around `tac.framework_agnostic.helpers.tinygrad_state_dict_to_npz_bridge` that PRODUCES a canonical ZIP-member with deterministic header per HNeRV parity L3 (monolithic 4-section archive grammar).
3. **Inflate-side numpy primitive consumer** — `load_tinygrad_trained_weights_for_numpy_inflate(archive_path, *, member_name)` that READS the ZIP-member + decompresses + returns `dict[str, numpy.ndarray]` ready for numpy-portable inflate primitives per HNeRV parity L4 (≤200 LOC + ≤2 deps + CUDA-or-CPU agnostic). **Zero tinygrad dependency at inflate time** — the inflate path is pure numpy per the bridge contract.
4. **Per-tensor metadata preservation** — `TinygradBridgeManifest` frozen dataclass carrying `tensor_count` + `total_uncompressed_bytes` + `compressed_bytes` + `per_tensor_shapes` + `per_tensor_dtypes` + canonical Provenance per Catalog #323.
5. **Framework-agnostic decorator wiring** — `@tinygrad_with_numpy_inflate_bridge` companion decorator (sister of `@mlx_first_with_numpy_fallback`) that resolves tinygrad-at-training + numpy-at-inflate routing in one declarative annotation.

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable + Catalog #290 design-memo discipline.

| Layer | Canonical helper available? | Decision | Rationale |
|---|---|---|---|
| Backend selection | `tac.framework_agnostic.backend.select_backend` (Catalog #205 sister) | **ADOPT_CANONICAL_BECAUSE_SERVES** | Backend.TINYGRAD already registered; this bridge just consumes it. Forking would re-introduce the duplicate-backend-selection anti-pattern this canonical helper extincts. |
| state_dict → npz bytes | `tac.framework_agnostic.helpers.tinygrad_state_dict_to_npz_bridge` | **ADOPT_CANONICAL_BECAUSE_SERVES** | Already routes tinygrad.Tensor → numpy.ndarray via `Tensor.numpy()` + `np.savez_compressed`. THIS module wraps it with ZIP-member packaging. |
| ZIP-member packaging | sister of PR95/PR100/PR101 monolithic 4-section archive grammar (HNeRV parity L3) | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | The canonical PR95 archive grammar packages MLX-trained weights; a tinygrad-trained substrate has its own per-substrate archive grammar choices. The bridge provides a CANONICAL ZIP-member ENCODING helper but each consuming substrate declares its own monolithic archive grammar per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD. |
| Inflate-side numpy consumer | `numpy.load` (canonical numpy oracle) | **ADOPT_CANONICAL_BECAUSE_SERVES** | numpy IS the canonical inflate-time framework per HNeRV parity L4 + Catalog #295 PYTHONPATH self-containment. The bridge wraps `numpy.load` with per-tensor metadata preservation. |
| Per-tensor metadata | `dataclasses.dataclass(frozen=True)` | **ADOPT_CANONICAL_BECAUSE_SERVES** | Sister of `AxisDecomposition` (Catalog #356) + `DeliverabilityProof` (Catalog #319) frozen-dataclass canonical pattern. |
| Canonical Provenance | `tac.provenance.build_provenance_for_predicted` (Catalog #323) | **ADOPT_CANONICAL_BECAUSE_SERVES** | Tinygrad bridge is observability-only training-time surface per Catalog #287/#323; non-promotable per Catalog #192/#317 (sister of MLX-research-signal pattern). |
| Decorator surface | `tac.framework_agnostic.decorators.mlx_first_with_numpy_fallback` + sister | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | The MLX-first decorator targets MLX-on-Apple-Silicon. The tinygrad decorator targets tinygrad-on-any-platform (Apple Silicon Metal + Linux CUDA + WebGPU + others). The canonical-priority-cascade differs, so the decorator forks. |
| Inflate runtime helper | `@inflate_runtime_helper` (existing canonical decorator) | **ADOPT_CANONICAL_BECAUSE_SERVES** | The inflate-side consumer should be wrapped with this canonical decorator to pin `Backend.NUMPY` per the bridge contract (HNeRV parity L4). |

## 9-dimension success checklist evidence

1. **UNIQUENESS** — tinygrad is the 3rd canonical training framework after MLX + PyTorch; class-shift surface IS the framework-portability axis (sister of cooperative-receiver / predictive-coding class-shifts on the substrate axis).
2. **BEAUTY + ELEGANCE** — ~150-250 LOC module mirroring `pr95_hnerv_mlx.py` structure at lighter scope (no training loop; this is bridge-only); PR101-style 30-sec-reviewable per HNeRV parity discipline.
3. **DISTINCTNESS** — explicitly different from sister `mlx_state_dict_to_npz_bridge` (which targets MLX) + sister `pytorch_state_dict_to_npz_bridge` (which targets PyTorch); tinygrad has unique runtime semantics (lazy evaluation; framework-agnostic compilation to METAL/CUDA/WebGPU/CLANG).
4. **RIGOR** — premise verification before edit (Catalog #229; canonical surface ALREADY exists in `tac.framework_agnostic.helpers.tinygrad_state_dict_to_npz_bridge`); adversarial review via STRICT preflight gates Catalog #335 (canonical contract) + Catalog #341 (Tier A canonical markers); per-test empirical anchor.
5. **OPTIMIZATION PER TECHNIQUE** — tinygrad-specific lazy evaluation realized via `Tensor.realize()` BEFORE `.numpy()` per tinygrad canonical pattern; sister of `pr95_hnerv_mlx.py::mx.eval()` discipline.
6. **STACK-OF-STACKS-COMPOSABILITY** — orthogonal to MLX bridge (different framework) + orthogonal to PyTorch bridge (different framework); composable with any inflate-time numpy primitive per HNeRV parity L4.
7. **DETERMINISTIC REPRODUCIBILITY** — npz round-trip byte-deterministic per `tac.framework_agnostic.helpers.npz_to_numpy_primitives` (canonical oracle); seed-pinned via `tinygrad.helpers.dtypes.from_np` parity.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — tinygrad's `Tensor.realize()` + canonical lazy-eval-then-materialize discipline; per-tensor metadata preservation surface ranks alongside per-axis decomposition (Catalog #356).
9. **OPTIMAL MINIMAL CONTEST SCORE** — N/A (this is the bridge SURFACE, not a substrate; promotion gates per Catalog #205/#295/#127/#192/#317 remain). The bridge enables NEW substrate trainers to choose tinygrad as their training-time framework while preserving the canonical numpy-portable inflate contract.

## Cargo-cult audit per assumption

Per Catalog #303 per-assumption HARD-EARNED-vs-CARGO-CULTED classification.

| Assumption | Classification | Rationale | Unwind path |
|---|---|---|---|
| "tinygrad is a viable training framework alternative to MLX + PyTorch" | **HARD-EARNED** | tinygrad is mature (15k+ GitHub stars; production use at George Hotz's tinycorp; cross-platform support METAL/CUDA/WebGPU/CLANG). | N/A (well-supported assumption). |
| "tinygrad's `Tensor.numpy()` is byte-deterministic" | **HARD-EARNED** | Per tinygrad source `tinygrad/tensor.py::Tensor.numpy` materializes via `realize()` then `np.frombuffer` from the canonical buffer; numerics are deterministic per `tinygrad.helpers.GlobalCounters.cache_collect` discipline. | N/A. |
| "the canonical bridge contract (state_dict → npz → ZIP-member → numpy inflate) generalizes from MLX to tinygrad" | **HARD-EARNED** | Per sister `tinygrad_state_dict_to_npz_bridge` IN helpers.py (already landed) — `np.savez_compressed` is framework-agnostic; the bridge is byte-deterministic regardless of source framework. | N/A. |
| "tinygrad's lazy evaluation requires explicit `Tensor.realize()` before `.numpy()`" | **HARD-EARNED** | Per tinygrad docs + source: `Tensor.numpy()` internally calls `Tensor.realize()` but explicit `realize()` before bulk-export is canonical for deterministic-timing measurement. | Sister of `pr95_hnerv_mlx.py::mx.eval()` discipline. |
| "the bridge needs a SEPARATE module (not just the existing helper)" | **HARD-EARNED** | Per cascade memo: 3rd sister surface requires NEW module sister of `pr95_hnerv_mlx.py`. The existing `tinygrad_state_dict_to_npz_bridge` HELPER is the canonical wrapped primitive; the NEW module IS the canonical ZIP-member-packaging + per-tensor-metadata-preservation + decorator surface. | N/A. |
| "tinygrad availability detection via `importlib.util.find_spec` is sufficient" | **HARD-EARNED** | Per canonical `Backend._find_spec_safe` + `_is_tinygrad_available` already landed in `tac.framework_agnostic.backend`. | N/A. |

## Observability surface

Per Catalog #305 observability declaration; 6 facets.

1. **Inspectable per layer** — `TinygradBridgeManifest` exposes per-tensor shape + dtype + compressed byte count for every export operation; downstream consumers (autopilot ranker, sensitivity map) can inspect bridge throughput per substrate.
2. **Decomposable per signal** — bridge throughput decomposes into `tinygrad_realize_ms` (tinygrad-side compute) + `tinygrad_to_numpy_ms` (framework boundary) + `npz_compression_ms` (numpy oracle) + `zip_packaging_ms` (archive grammar).
3. **Diff-able across runs** — bridge manifests are JSON-serializable via `manifest_to_dict()` so two runs of the same substrate can be diff-ed at compressed-bytes + per-tensor-shape level.
4. **Queryable post-hoc** — manifests written to `experiments/results/<lane>/tinygrad_bridge_manifest.json` (sister of `pr95_hnerv_mlx.py` checkpoint discipline); CLI consumer at `tools/audit_tinygrad_bridge_throughput.py` (planned phase 6).
5. **Cite-able** — every manifest carries canonical Provenance per Catalog #323 (commit_sha + lane_id + call_id + UTC); auto-discovered via Catalog #335 cathedral consumer auto-discovery (planned phase 6).
6. **Counterfactual-able** — sister of byte-mutation smoke per Catalog #105/#139; "what if this byte changed?" probes operate on the inflate-side numpy primitive layer (downstream of the bridge) — bridge changes propagate deterministically per the canonical oracle.

## Predicted ΔS band

**N/A** — this is the BRIDGE surface, not a substrate. The bridge enables NEW substrate trainers to choose tinygrad without per-substrate rediscovery of the canonical bridge contract; the score impact materializes per-substrate when a substrate-engineering trainer adopts tinygrad and dispatches via the canonical 4-tier paired-CUDA + CPU contest auth-eval per Catalog #246. Dykstra-feasibility check per Catalog #296 N/A for this layer — the bridge layer is observability-only per Catalog #287/#323.

`# PREDICTED_BAND_VIBES_OK:bridge-surface-no-substrate-no-score-prediction-per-canonical-helper-surface-vs-substrate-distinction`

## Horizon class

**plateau_adjacent** (per Catalog #309). The bridge enables substrate trainers but does NOT itself shift score class. Future substrate-class-shift work using tinygrad (e.g., a class-shift NeRV variant trained on tinygrad-Metal on M5 Max) would carry `horizon_class: frontier_pursuit` or `asymptotic_pursuit` independently.

`horizon_class: plateau_adjacent`

## Class-shift not bolt-on

**N/A** (sister of Catalog #310). The bridge surface is FRAMEWORK-PORTABILITY, not a substrate class-shift. Substrate class-shift architectures consume the bridge (when their author chooses tinygrad over MLX) but the bridge itself is class-portability infrastructure.

## Drift surface declaration

Per `feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md` (sister standing directive). The 5 canonical drift sources apply at the FRAMEWORK BOUNDARY (tinygrad → numpy):

1. **Bfloat16/fp16 precision** — tinygrad supports bfloat16 + float16; bridge MUST `.cast(dtypes.float32)` before `.numpy()` to preserve byte-identity with PR101 medal-class fp32 convention. **Drift surface: ACTIVE.**
2. **Softmax-LSE epsilon** — N/A at bridge layer (substrate-trainer responsibility).
3. **AdamW β state** — N/A at bridge layer; optimizer state is per-substrate-trainer.
4. **Bicubic non-bit-identity** — N/A at bridge layer; bridge does NOT resize tensors.
5. **EMA Kahan precision** — N/A at bridge layer; EMA shadow is per-substrate-trainer (Catalog #88 sister).

**Mitigation:** explicit `.cast(dtypes.float32).realize().numpy()` cascade in the bridge helper; the canonical numpy oracle (npz round-trip) IS the byte-determinism guarantee per Catalog #146.

## Implementation outline (Phase B detail)

### Module structure (`src/tac/local_acceleration/tinygrad_bridge.py`)

~150-250 LOC per HNeRV parity L4 budget. Sections:

1. **SPDX header + module docstring + canonical cross-references** (~30 LOC)
2. **Deferred-import guards** — `_TINYGRAD_IMPORT_ERROR` per canonical `pr95_hnerv_mlx.py` pattern (~15 LOC)
3. **Canonical constants** — `LANE_ID` / `TINYGRAD_BRIDGE_SCHEMA` / `BRIDGE_MANIFEST_SCHEMA` (~10 LOC)
4. **`TinygradBridgeManifest` frozen dataclass** — `__post_init__` invariants per Catalog #356 sister pattern (~30 LOC)
5. **`tinygrad_state_dict_to_zip_member_bytes(...)` canonical bridge** — wraps `tac.framework_agnostic.helpers.tinygrad_state_dict_to_npz_bridge` + packages as ZIP-member with deterministic header (~30 LOC)
6. **`load_tinygrad_trained_weights_for_numpy_inflate(...)` inflate-side consumer** — `@inflate_runtime_helper` decorated for `Backend.NUMPY` pinning per HNeRV parity L4 (~30 LOC)
7. **`build_tinygrad_bridge_manifest(...)` operator-facing manifest builder** — per-tensor metadata + canonical Provenance per Catalog #323 (~30 LOC)
8. **`@tinygrad_with_numpy_inflate_bridge` decorator** — sister of `@mlx_first_with_numpy_fallback`; resolves tinygrad-at-training + numpy-at-inflate routing (~30 LOC)
9. **`__all__` declaration** (~10 LOC)

### Tests (`src/tac/tests/test_tinygrad_portable_inflate_primitive_bridge.py`)

~15-25 tests covering:

1. **Availability detection** — `is_tinygrad_available()` returns True/False per env
2. **Bridge contract round-trip** — `state_dict → bridge bytes → numpy primitives` byte-deterministic
3. **ZIP-member packaging** — deterministic header per HNeRV parity L3 monolithic archive grammar
4. **Inflate-side consumer** — numpy-only path; no tinygrad import at inflate time
5. **Catalog #295 PYTHONPATH self-containment** — inflate consumer survives empty PYTHONPATH per canonical contract
6. **Per-tensor metadata preservation** — `TinygradBridgeManifest` invariants
7. **Canonical Provenance** — manifest carries `score_claim=False` + `axis_tag=[predicted]` + non-promotable markers per Catalog #287/#323
8. **Framework-agnostic decorator routing** — `@tinygrad_with_numpy_inflate_bridge` resolves correctly
9. **Drift surface** — bfloat16 → fp32 cast verified byte-deterministic
10. **HNeRV parity L4 budget** — module LOC <= 250 (substrate_engineering exception per L7)

### Canonical equation registration

If empirical anchor surfaces during PHASE C testing (e.g. tinygrad-bridge throughput vs MLX-bridge throughput on M5 Max), register canonical equation per Catalog #344:
- `tinygrad_portable_inflate_primitive_bridge_byte_determinism_v1` (if byte-determinism invariant surfaces as new mathematical pattern beyond existing sister `framework_agnostic_backend_abstraction_compounding_v1`)

PHASE D will REGISTER OR DEFER per empirical findings.

## Cathedral consumer auto-discovery

Per Catalog #335 canonical contract: the bridge is observability-only per Catalog #341 Tier A. The canonical `cathedral_equation_lookup_consumer` (sister of `mps_viable_prescreen_consumer`) inherits auto-discovery; this module does NOT land a NEW cathedral consumer because the bridge surface is consumed via the existing framework-agnostic cathedral consumer `tac.cathedral_consumers.framework_agnostic_lookup_consumer` (sister consumer that already exists per cathedral_consumers/ listing).

## 6-hook wire-in declaration (Catalog #125)

- **hook #1 sensitivity-map** = N/A (bridge-surface; sensitivity is per-substrate)
- **hook #2 Pareto constraint** = N/A (bridge-surface; constraints are per-substrate)
- **hook #3 bit-allocator** = N/A (bridge-surface; bit allocation is per-substrate)
- **hook #4 cathedral autopilot dispatch** = ACTIVE via existing `tac.cathedral_consumers.framework_agnostic_lookup_consumer` sister consumer (auto-discovery per Catalog #335 inherits)
- **hook #5 continual-learning posterior** = ACTIVE (bridge manifests written to `experiments/results/<lane>/tinygrad_bridge_manifest.json` are queryable canonical posterior anchors via sister CLI `tools/audit_tinygrad_bridge_throughput.py`)
- **hook #6 probe-disambiguator** = N/A (no 2+ defensible interpretations at bridge surface)

## Lane registry declaration

```yaml
lane_id: lane_slot_j_cascade_item_6_tinygrad_portable_inflate_primitive_bridge_20260529
level: 1
lane_class: substrate_engineering  # per HNeRV parity L7; bridge IS infrastructure
research_only: true                # per Catalog #295/#220; bridge surface only
notes: |
  Tinygrad-portable inflate primitive bridge per cascade item 6 of
  7-item operator-bound cascade. 3rd sister surface per MLX-FIRST 8th
  standing directive (MLX bridge + numpy fallback + tinygrad bridge).
  archive_grammar=numpy_npz_zip_member_with_canonical_header
  parser_section_manifest=TinygradBridgeManifest_with_per_tensor_metadata
  inflate_runtime_loc_budget=200
  runtime_dep_closure=numpy,zipfile
  export_format=numpy_npz_inside_zip_member
  score_aware_loss=N/A_bridge_surface
  bolt_on_loc_budget=350
  no_op_detector_planned=true
  score_improvement_mechanism_status=BRIDGE_INFRASTRUCTURE_ONLY
  horizon_class=plateau_adjacent
```

## Cross-references

- CLAUDE.md "MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL" 8th standing directive (parent canonical contract)
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L4 (≤200 LOC + ≤2 ext deps inflate budget)
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" (per-substrate fork-vs-canonical decision)
- CLAUDE.md "Bit-level deconstruction and entropy discipline" (npz byte-deterministic oracle)
- `feedback_cathedral_autopilot_is_the_canonical_meta_orchestrator_proceed_with_all_7_cascade_20260528.md` (cascade item 6 source)
- `feedback_mlx_first_numpy_portable_individually_fractally_optimized_standing_directive_20260526.md` (8th standing directive canonical reference)
- `feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md` (drift surface declaration sister)
- `src/tac/local_acceleration/pr95_hnerv_mlx.py` (MLX bridge reference pattern; THIS module is the tinygrad sister surface)
- `src/tac/framework_agnostic/helpers.py::tinygrad_state_dict_to_npz_bridge` (canonical bridge primitive this module wraps)
- `src/tac/framework_agnostic/backend.py::Backend.TINYGRAD` (canonical Backend taxonomy)
- `src/tac/framework_agnostic/operations.py::_quantize_int8_tinygrad` (canonical tinygrad operations sister)
- `src/tac/substrates/_shared/inflate_runtime.py::select_inflate_device` (Catalog #205 canonical inflate-time device-selection sister)
- Catalog #146 / #205 / #220 / #270 / #272 / #287 / #290 / #294 / #295 / #303 / #305 / #309 / #310 / #323 / #335 / #341 / #344 / #356 / #357
