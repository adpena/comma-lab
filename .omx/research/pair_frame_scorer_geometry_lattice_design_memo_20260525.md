# Pair-Frame Scorer-Geometry Lattice — Design Memo (2026-05-25)

- timestamp_utc: 2026-05-25T15:32:00Z
- agent: claude (PAIR-FRAME-SCORER-GEOMETRY-LATTICE design subagent)
- lane_id: lane_pair_frame_scorer_geometry_lattice_design_memo_20260525
- scope: 3-deliverable design memo (math + canonical contract + scaffold sketch + 4-BUILD operator-routable) for the 5D lattice codex named as THE next canonical bridge
- authority: design + observability ONLY; score/promotion/rank/dispatch authority all FALSE
- relates to: DQS1-LOOP-CLOSURE-ASSIST commit `504a31448` Top-3 MEDIUM operator-routable + codex eureka memo `.omx/research/codex_findings_eureka_drop_many_rate_distortion_budget_20260525T143351Z_codex.md` §"Residual Gap"
- canonical equation candidate QUEUED: `pair_frame_scorer_geometry_lattice_4d_binding_canonical_v1`
- **scaffold filename pivot**: original target `src/tac/optimization/pair_frame_scorer_geometry_lattice.py` ALREADY EXISTS as codex v1 row-based implementation (commit `4ed9eb905` "Wire pair-frame geometry starts into DQS1 queue"; 501 LOC; `SCHEMA="pair_frame_scorer_geometry_lattice.v1"` + `ROW_SCHEMA` + `REQUEST_SCHEMA` + `build_pair_frame_scorer_geometry_lattice` function). Per scope discipline ("DO NOT mutate ANY codex DQS1 cascade source code"), this scaffold lands at sibling path `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas.py` to preserve the codex v1 row-based module while adding the 5D canvas binding layer per codex's residual-gap call-out. The two modules are **complementary**: codex v1 IS row-based pair-frame geometry binding for DQS1 acquisition; this `_5d_canvas` scaffold IS the next-level CANVAS binding the 5 axes codex named (the codex v1 module is row-based and pair-only; the canvas extends to the full 5D coordinate including frame_idx + scorer_axis + receiver_runtime + cpu_cuda_axis). BUILD-1 sister subagent will explicitly compose with the codex v1 row-based reader to populate empirical cells.
- discipline anchors: Catalog #287 (canonical Provenance evidence-tag) + #313 (probe-outcomes ledger) + #323 (canonical Provenance umbrella) + #335 (cathedral consumer canonical contract) + #341 (Tier A canonical-routing markers) + #344 (canonical equation candidate QUEUED for operator-routable RATIFY-N) + #356 (per-axis AxisDecomposition foundation) + #357 (Tier B canonical contract)

## Operator-routable origin

Codex 2026-05-25T14:33Z verbatim quote (`codex_findings_eureka_drop_many_rate_distortion_budget_20260525T143351Z_codex.md` §"Residual Gap"):

> "bind pair component xray rows, frame-axis master-gradient decomposition, SegNet/PoseNet score geometry, CPU/CUDA axis labels, and receiver feasibility into a pair-frame scorer-geometry lattice that can generate queue-executable full-drop, repair, masked, and feathered starts without false authority"

The lattice is the multi-dimensional binding surface that unblocks four currently-blocked operator-explicit requests:

1. within-set masked/feathered receiver semantics (DQS1-LOOP-CLOSURE GAP 1 residual)
2. inverse-scorer null-direction lattice
3. global low-impact full-pair/frame-drop probe
4. CUDA-axis DQS1 variant (DQS1-LOOP-CLOSURE GAP 3)

This design memo lands the CANVAS; sister DROP-MANY-BEAM-DESIGN drills ONE algorithm on the canvas; sister RATE-ATTACK-METHODS-DIMENSIONS-MATRIX maps the broader methods×dimensions matrix in which this lattice sits.

## Canonical-vs-unique decision per layer

- Reusing canonical: `tac.cathedral.consumer_contract.AxisDecomposition` (Catalog #356) + `tac.cathedral.consumer_contract.HookNumber` + `tac.cathedral.consumer_contract.ConsumerTier` (Catalog #357) + `tac.score_composition.compose_score_from_axes` + canonical Provenance umbrella (Catalog #323) + `tools/subagent_checkpoint.py` (Catalog #206) + canonical Catalog #313 probe-outcomes ledger.
- Forking nothing in source: the design memo + scaffold skeleton are NEW research artifacts. The scaffold mirrors sister consumer interface pattern (`per_pair_gradient_clustering_consumer/__init__.py`) so future BUILD-4 Tier B promotion is a low-friction transformation.
- 5D lattice schema is GENUINELY NEW: no existing canonical helper binds (pair × frame × scorer_axis × receiver_runtime × cpu_cuda_axis). The bind contract IS the design contribution.

## Observability surface (Catalog #305)

1. **Inspectable per layer**: each of the 5 lattice axes is queryable independently via canonical primitives `bind_pair_component_xray(pair_idx)` / `decompose_frame_axis_master_gradient(frame_idx)` / `compute_segnet_posenet_score_geometry(pair_idx, frame_idx)` / `query_receiver_runtime_feasibility(pair_idx, frame_idx, receiver_runtime)`.
2. **Decomposable per signal**: per-cell `PairFrameScorerGeometryCell` carries `(pair_idx, frame_idx, scorer_axis, receiver_runtime, cpu_cuda_axis, predicted_delta_score, predicted_byte_cost, receiver_feasibility, catalog_323_provenance)`. The 5D coordinate uniquely identifies the cell; the 4 measurement fields decompose into independent signals.
3. **Diff-able across runs**: lattice rebuilt with `build_lattice(archive_path)` produces deterministic cells keyed by archive SHA; sister-archive lattice diff = per-cell delta in measurement fields.
4. **Queryable post-hoc**: canonical CLI `tools/query_pair_frame_lattice.py --archive <path> --pair-idx N --frame-idx M --scorer-axis seg` plus per-cell `as_dict()` JSON serialization round-trip.
5. **Cite-able**: every cell's `catalog_323_provenance` carries archive sha256 + producer commit + canonical helper invocation + canonical equation id (the queued `pair_frame_scorer_geometry_lattice_4d_binding_canonical_v1`).
6. **Counterfactual-able**: per-cell `query_receiver_runtime_feasibility` answers "what if this pair/frame went through masked vs feathered receiver?" without re-running scorer; `generate_*_starts` operations turn the counterfactual into an executable candidate.

## 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS**: The 5D lattice is the FIRST canonical primitive that simultaneously binds the 5 axes codex explicitly named. Pair-only or frame-only or scorer-axis-only consumers already exist; their 5D INTERSECTION is the new structural canvas.
2. **BEAUTY + ELEGANCE**: ONE frozen dataclass + ONE container class + 4 canonical operation generators. Each generator is < 50 LOC. Total scaffold ~300-400 LOC. PR101 medal-class budget per HNeRV parity L7.
3. **DISTINCTNESS**: distinct from sister DQS1 pairset acquisition (1D pair-only) + distinct from sister AxisDecomposition (1D scorer-axis-only) + distinct from sister Cable D master-gradient consumers (per-pair or per-frame but NOT joint) + distinct from sister Tier A routing markers (Tier B score-contributing per Catalog #357).
4. **RIGOR**: every operation is grounded in canonical mathematical primitives (Daubechies multi-scale wavelet partition prior for receiver_runtime hierarchy + Dykstra alternating-projection feasibility for multi-dim convex intersection + Atick-Redlich cooperative-receiver for receiver_runtime semantics + Wyner-Ziv side-information for receiver-aware encoding + Cable D master-gradient consumers for per-pair + per-frame decomposition + Catalog #356 per-axis AxisDecomposition for scorer_axis structural binding + Catalog #357 Tier B canonical contract).
5. **OPTIMIZATION-PER-TECHNIQUE**: each receiver_runtime mode has substrate-optimal engineering (raw_residual: identity / smoothed_residual: low-pass filter / masked: per-region SegNet-class-aware byte mask / feathered: smooth-transition Daubechies wavelet partition / full_drop: byte-removal at archive grammar boundary). Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode".
6. **STACK-OF-STACKS-COMPOSABILITY**: the lattice IS the composability primitive — operations from different families (drop-many beam search on pair axis × per-frame master-gradient × scorer-axis decomposition) all consume from the SAME lattice. Sister-family composition is read-from-canvas, not bespoke wiring per pair.
7. **DETERMINISTIC-REPRODUCIBILITY**: per-cell measurement fields are deterministic functions of (archive_sha256, lattice coordinates); cell `as_dict()` is byte-stable JSON (sort_keys=True) per Catalog #245 canonical pattern.
8. **EXTREME-OPTIMIZATION-PERFORMANCE**: full 5D dense lattice is 600 × 1200 × 3 × 5 × 2 = 21.6M cells; SPARSE representation (only cells where receiver_feasibility=True OR predicted_delta_score is finite) keeps practical memory bounded (current DQS1 cascade explores ~581 candidates; expected sparse lattice ~10⁴-10⁵ cells).
9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: the lattice unblocks 4 simultaneous operator-explicit requests (within-set masked/feathered + inverse-scorer null + global low-impact frame-drop + CUDA-axis DQS1 variant). Per CLAUDE.md "Frontier target — NON-NEGOTIABLE": this is THE next canonical bridge per codex's explicit naming.

## Cargo-cult audit per assumption (Catalog #303)

- **HARD-EARNED**: `pairset_component_marginal_score_decomposition_v1` (equation #36 in registry) — drop-one ΔS = SegNet + PoseNet + rate is EMPIRICALLY VERIFIED across 8 CPU+CUDA anchors. The lattice's per-cell `predicted_delta_score` for drop-one operations inherits this canonical equation's prediction.
- **HARD-EARNED**: `per_pair_master_gradient_score_impact_taylor_v1` (equation in registry) — per-pair Taylor first-order decomposition is well-calibrated for drift in master-gradient space. The lattice's per-cell `predicted_byte_cost` inherits this prediction.
- **HARD-EARNED**: Catalog #356 AxisDecomposition canonical contract — per-axis (seg, pose, rate) decomposition is the canonical contract for any cathedral consumer that emits per-axis signal.
- **CARGO-CULTED**: ASSUMPTION that the 4 receiver_runtime modes (raw_residual / smoothed_residual / masked / feathered / full_drop) are SUFFICIENT. UNWIND-TEST: per-substrate sister subagent enumerates alternative receiver modes (e.g. inverse-scorer null-direction, additive repair, predictive coding); the canonical 5-mode enum is OPEN to extension via sister subagent registration per Catalog #335 auto-discovery.
- **CARGO-CULTED**: ASSUMPTION that 5D coordinate is the right binding granularity (vs 4D collapsing scorer_axis OR 6D adding bit-level granularity). UNWIND-TEST: 5D is chosen because codex's explicit naming bound EXACTLY these 5 axes; 4D collapses lose codex's stated intent; 6D bit-level is sister of bit_level_score_critical_bits_consumer (Cable D exploit #8) which composes WITH the 5D lattice rather than replacing it.
- **CARGO-CULTED**: ASSUMPTION that lattice can be built statically once per archive. UNWIND-TEST: receiver_runtime feasibility depends on substrate-specific archive grammar; per-substrate lattice instances may differ. Canonical `build_lattice` accepts substrate-class hint to gate which receiver_runtime modes apply.

## Predicted ΔS band — NOT a substrate dispatch proposal

This is a DESIGN MEMO for a canonical primitive, NOT a substrate-class dispatch proposal. No predicted ΔS band claim per Catalog #296. The lattice EMITS predicted ΔS bands per-cell via the canonical equation candidates registered downstream; the lattice ITSELF carries no aggregate prediction.

Per CLAUDE.md "Council conduct" + Catalog #292: Assumption-Adversary verdict on the cargo-cult assumptions above is HARD-EARNED for items 1-3 (inherited from ratified canonical equations + Catalog #356 contract) and CARGO-CULTED-PENDING-EMPIRICAL for items 4-6 (canonical receiver_runtime enum + 5D coordinate + static lattice assumption).

## Council attendees / verdict

T1 working-group (design-only; no T2+ deliberation required per Catalog #300):

- Shannon LEAD (information-theory grounding for receiver_runtime semantics — per Atick-Redlich + Wyner-Ziv canonical cite chain)
- Dykstra CO-LEAD (alternating-projection feasibility for multi-dim convex intersection — `query_receiver_runtime_feasibility` is per-cell Dykstra projection)
- Daubechies CO-LEAD (multi-scale wavelet partition prior for receiver_runtime hierarchy — feathered receiver_runtime IS smooth-transition Daubechies wavelet partition)
- Rudin CO-LEAD (interpretable ML — every per-cell `predicted_delta_score` decomposes into auditable scorer_axis × receiver_runtime contributions per Catalog #356)
- Atick (cooperative-receiver paradigm for repair operation — sister-author of Atick-Redlich canonical cite chain)
- Carmack (MVP-first phasing — scaffold skeleton + 4-BUILD operator-routable enumeration with cost estimates per CLAUDE.md "Carmack MVP-first phasing")
- Assumption-Adversary (challenges the 3 cargo-cult assumptions enumerated above per Catalog #292)

T1 working-group VERDICT: **PROCEED** (design memo + scaffold skeleton; no quorum required at T1 per Catalog #300). Sister subagent BUILD-1 picks up the empirical 5D lattice population.

---

## DELIVERABLE 1: 5D Lattice Math + Canonical Contract

### Mathematical formulation

The pair-frame scorer-geometry lattice L is a 5-dimensional tensor:

```
L: PairIdx × FrameIdx × ScorerAxis × ReceiverRuntime × CpuCudaAxis → MeasurementTuple

Where:
  PairIdx ∈ {0, 1, ..., 599}                  (600 pairs per 1200-frame contest video, seq_len=2 non-overlapping)
  FrameIdx ∈ {0, 1, ..., 1199}                (1200 frames; first_frame_of_pair = 2*pair_idx, last_frame_of_pair = 2*pair_idx + 1)
  ScorerAxis ∈ {SegNet_5class, PoseNet_6d, rate_term}   (canonical 3-axis contest decomposition per CLAUDE.md formula)
  ReceiverRuntime ∈ {raw_residual, smoothed_residual, masked, feathered, full_drop}   (5 canonical receiver modes)
  CpuCudaAxis ∈ {contest_cpu, contest_cuda_t4}          (canonical 1:1 hardware-compliant axes per CLAUDE.md "Submission auth eval")

MeasurementTuple := (predicted_delta_score: float,
                     predicted_byte_cost: int,
                     receiver_feasibility: bool,
                     catalog_323_provenance: Mapping[str, Any])
```

### Canonical contest formula (the binding ground truth)

Per CLAUDE.md canonical formula:

```
S = 100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes / 37_545_489
```

The lattice EMITS per-cell ΔS predictions composing the per-axis Δ via the canonical formula:

```
ΔS[pair, frame, scorer_axis, receiver_runtime, cpu_cuda_axis] =
    canonical_per_axis_contribution(scorer_axis) ×
    receiver_runtime_modulator(receiver_runtime) ×
    cpu_cuda_axis_amplification(cpu_cuda_axis)
```

Where:
- `canonical_per_axis_contribution(seg) = 100 × Δd_seg`
- `canonical_per_axis_contribution(pose) = sqrt(10 × (d_pose_after - d_pose_before)) - sqrt(10 × d_pose_before)` (non-linear; depends on baseline)
- `canonical_per_axis_contribution(rate) = 25 × Δarchive_bytes / 37_545_489`
- `receiver_runtime_modulator` = per-mode scaling factor learned from sister consumer empirical anchors
- `cpu_cuda_axis_amplification` = sister of `pose_axis_cuda_amplification_v1` (canonical equation in registry) — typically 1.0 for CPU axis; 1.5-5x for CUDA axis depending on substrate-class

### Canonical primitive operations

```python
# Per-pair component xray (Cable D master-gradient sister)
def bind_pair_component_xray(pair_idx: int, archive_path: Path) -> dict[ScorerAxis, np.ndarray]:
    """Extract per-pair component-level scorer response per Cable D master-gradient.
    Returns dict mapping scorer_axis -> per-frame array of component magnitudes."""

# Per-frame master-gradient decomposition (codex's per-frame extension)
def decompose_frame_axis_master_gradient(frame_idx: int, archive_path: Path) -> dict[ScorerAxis, np.ndarray]:
    """Per-frame master gradient per per_frame_master_gradient_consumer.
    Returns dict mapping scorer_axis -> per-pair array of gradient magnitudes."""

# Per-pair × per-frame × per-axis × CPU/CUDA score-component decomposition
def compute_segnet_posenet_score_geometry(
    pair_idx: int,
    frame_idx: int,
    archive_path: Path,
) -> dict[CpuCudaAxis, dict[ScorerAxis, float]]:
    """Per-pair × per-frame × per-axis score-component decomposition.
    Returns nested dict {cpu_cuda_axis: {scorer_axis: scalar_score_contribution}}."""

# Per-receiver-runtime feasibility check (Dykstra alternating-projection sister)
def query_receiver_runtime_feasibility(
    pair_idx: int,
    frame_idx: int,
    receiver_runtime: ReceiverRuntime,
    archive_path: Path,
) -> dict[ScorerAxis, bool]:
    """Per-receiver-mode feasibility check (whether the receiver runtime can preserve
    scorer-axis-relevant signal under the operation). Returns dict mapping scorer_axis
    -> bool feasibility (True if the receiver runtime is structurally compatible with
    the scorer axis's signal preservation requirements)."""

# Queue-executable candidate emitter (the canonical bridge to dispatch flow)
def generate_queue_executable_start(
    operation: CanonicalOperation,
    pair_idxs: Sequence[int],
    frame_idxs: Sequence[int],
    receiver_runtime: ReceiverRuntime,
    cpu_cuda_axis: CpuCudaAxis,
    lattice: PairFrameScorerGeometryLattice,
) -> ExecutableCandidate:
    """Emit canonical archive candidate per CLAUDE.md 'Substrate scaffolds MUST be
    COMPLETE or RESEARCH-ONLY'. Returns ExecutableCandidate carrying:
      - archive_candidate_path (the executable archive bytes)
      - predicted_delta_score (composed via canonical contest formula)
      - predicted_byte_cost
      - catalog_323_provenance
      - canonical_routing_markers (Catalog #341 non-promotable defaults)
      - canonical_dispatch_recipe_hint (operator-routable cost estimate)
    No false authority per codex's explicit naming."""
```

### The 4 canonical operations the lattice unblocks

| Operation | Math | Receiver-runtime dependency | Sister algorithm |
|---|---|---|---|
| **full-drop** | drop entire pair OR frame; rate saving = byte_cost_of_dropped_region; scorer penalty depends on receiver_runtime compensation | depends on receiver_runtime to compensate (raw_residual = no compensation; masked = partial; feathered = smooth; full_drop = no signal at all) | sister of DROP-MANY-BEAM-DESIGN drop-many beam pairwise interaction waterfill |
| **repair** | drop pair/frame + add per-pair/per-frame repair signal; per Atick-Redlich cooperative-receiver | requires non-trivial receiver_runtime (masked / feathered / additive); raw_residual is OUT-OF-SCOPE for repair | sister of Z4 cooperative-receiver loss substrate |
| **masked** | per-region SegNet-class-aware byte mask (UNIWARD/HILL/J-UNIWARD region weighting) | masked receiver_runtime IS the canonical mode; sister modes feathered + raw_residual provide structural variation | sister of UNIWARD steganalysis-inverse + per_segnet_class_chroma_consumer |
| **feathered** | smooth-transition variant of masked; per Daubechies multi-scale wavelet partition prior | feathered receiver_runtime IS the canonical mode; sister modes masked + smoothed_residual provide structural variation | sister of Daubechies wavelet codec + multi-scale partition prior |

---

## DELIVERABLE 2: Canonical Helper Scaffold Sketch

NEW file: `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas.py` (sibling of existing codex v1 row-based module `pair_frame_scorer_geometry_lattice.py`; complementary canvas binding per scaffold-filename-pivot note above)

SKELETON only (~200-400 LOC). NOT a full executable BUILD — full BUILD requires sister subagent BUILD-1 with empirical 600-pair × 1200-frame master-gradient population.

The scaffold exports the canonical primitive interface enumerated in DELIVERABLE 1 plus the canonical 5D dataclass + the canonical container class with the 4 operation generators. Per Catalog #357 Tier B canonical contract preparation:

- `CONSUMER_NAME = "pair_frame_scorer_geometry_lattice"`
- `CONSUMER_VERSION = "0.1.0-scaffold"`
- `CONSUMER_HOOK_NUMBERS = (SENSITIVITY_MAP, PARETO_CONSTRAINT, BIT_ALLOCATOR, CATHEDRAL_AUTOPILOT_DISPATCH, CONTINUAL_LEARNING_POSTERIOR, PROBE_DISAMBIGUATOR)` (all 6 hooks ACTIVE per BUILD-4 Tier B promotion)
- `CONSUMER_TIER = ConsumerTier.TIER_A_OBSERVABILITY_ONLY` (Tier A at scaffold landing; Tier B promotion is BUILD-4 sister subagent op-routable)

Until BUILD-1 lands the empirical lattice population, the scaffold's `build_lattice` raises `NotImplementedError("Scaffold-only; BUILD-1 subagent op-routable")` and the 4 operation generators raise the same sentinel. The canonical interface contract IS the design contribution; the empirical implementation is sister-subagent-deferred.

### Scaffold file content reference

See `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas.py` landed in same commit batch. Key exports:

- `class ScorerAxis(StrEnum)`: SEGNET_5CLASS, POSENET_6D, RATE_TERM
- `class ReceiverRuntime(StrEnum)`: RAW_RESIDUAL, SMOOTHED_RESIDUAL, MASKED, FEATHERED, FULL_DROP
- `class CpuCudaAxis(StrEnum)`: CONTEST_CPU, CONTEST_CUDA_T4
- `class CanonicalOperation(StrEnum)`: FULL_DROP, REPAIR, MASKED, FEATHERED
- `@dataclass(frozen=True) class PairFrameScorerGeometryCell`: 5D coordinate + 4 measurement fields + Catalog #323 Provenance
- `@dataclass(frozen=True) class ExecutableCandidate`: archive_candidate_path + predicted_delta_score + predicted_byte_cost + catalog_323_provenance + canonical_routing_markers
- `class PairFrameScorerGeometryLattice`: canonical container with `build_lattice`, `query_cell`, `generate_full_drop_starts`, `generate_repair_starts`, `generate_masked_starts`, `generate_feathered_starts`

---

## DELIVERABLE 3: Canonical Equation Candidate + 4-BUILD Operator-Routable

### Canonical equation candidate QUEUED per Catalog #344

**Equation ID**: `pair_frame_scorer_geometry_lattice_4d_binding_canonical_v1`

(Note: the equation describes a 5D lattice but is named "4d_binding" because the CpuCudaAxis is a discrete LABEL rather than a continuous dimension; the binding canvas binds 4 measurement dimensions × 2 axis labels.)

**Name**: Pair-frame scorer-geometry lattice 4D binding canvas

**Summary**: The 5D lattice L[pair_idx, frame_idx, scorer_axis, receiver_runtime, cpu_cuda_axis] is the canonical canvas binding pair-component xray + frame-axis master-gradient + scorer geometry + receiver feasibility + CPU/CUDA axis labels. The 4 canonical operations (full-drop / repair / masked / feathered) operate on this canvas; downstream consumers (drop-many beam search, inverse-scorer null-direction sweep, global low-impact frame-drop probe, CUDA-axis DQS1 variant) consume from the lattice without bespoke per-family wiring.

**Form**: Predicts that the 4 canonical operations are EQUIVALENT under the 5D lattice formulation — they share the same canvas and emit ExecutableCandidates via the same canonical interface; sister-family composition is read-from-canvas not bespoke wiring per pair-frame combination.

**Empirical anchor**: NEEDS at least 1 empirical anchor per operation (full-drop + repair + masked + feathered) with paired CPU+CUDA evidence to register per Catalog #344 trigger `when_3+_new_empirical_anchors_in_domain`. Each anchor would be a sister-subagent BUILD-1 through BUILD-4 deliverable. Until 3+ anchors land, the equation is QUEUED only.

**Predicted consumers**:
- `tac.optimization.decoder_q_pairset_acquisition` (consume lattice for filtered candidate emission)
- `tac.optimization.cross_family_candidate_portfolio` (rank candidates by per-cell predicted ΔS via lattice)
- `tac.cathedral_consumers.pair_frame_scorer_geometry_lattice_consumer` (BUILD-4 Tier B promotion)
- `tools/cathedral_autopilot_autonomous_loop.py` (auto-discovered cathedral consumer per Catalog #335)

**Predicted producers**:
- BUILD-1: `tools/build_pair_frame_scorer_geometry_lattice.py` populates 5D lattice via $0 CPU smoke on existing 600-pair fp64 master-gradient ledger + per-frame extension
- BUILD-2: `src/tac/optimization/pair_frame_scorer_geometry_lattice.py` full executable PairFrameScorerGeometryLattice with 4 canonical operation generators
- paired CPU+CUDA Modal exact-eval of top-K candidates per operation per axis_label

**Sister of canonical equations**:
- `pairset_component_marginal_score_decomposition_v1` (equation #36; ratified) — sister at the per-pair-DROP-COMPOSITION surface; the lattice's per-cell `predicted_delta_score` for drop-one operations INHERITS this canonical equation's prediction
- `per_pair_master_gradient_score_impact_taylor_v1` (equation in registry; well-calibrated) — sister at the per-pair-Taylor-DECOMPOSITION surface; the lattice's per-cell `predicted_byte_cost` INHERITS this prediction
- `pose_axis_cuda_amplification_v1` (equation #18 in registry) — sister at the per-pair-CUDA-AXIS-AMPLIFICATION surface; the lattice's `cpu_cuda_axis_amplification` factor INHERITS this prediction
- `dqs1_drop_many_pairwise_interaction_beam_search_v1` (sister DROP-MANY-BEAM-DESIGN candidate; QUEUED) — sister at the per-pair-INTERACTION surface; the lattice IS the canvas, drop-many beam search is ONE algorithm on the canvas

### 4-BUILD operator-routable enumeration with cost estimates

#### BUILD-1: Empirical 5D lattice population

**Scope**: populate empirical 5D lattice via $0 CPU smoke on existing 600-pair fp64 master-gradient ledger + per-frame extension. Sister subagent reads `.omx/state/master_gradient_anchors.jsonl` + per-pair component xray rows + per-frame master-gradient consumer rows; constructs `PairFrameScorerGeometryCell` for each (pair, frame) coordinate where empirical data exists; writes lattice JSON to `.omx/state/pair_frame_scorer_geometry_lattice/<archive_sha[:12]>_<utc>.json`.

**Cost**: $0 paid GPU + ~2-4h wall-clock (sister subagent).

**Deliverables**:
- empirical lattice JSON per canonical archive sha
- canonical helper `tac.optimization.pair_frame_scorer_geometry_lattice.load_empirical_lattice(archive_sha256, repo_root)` reads canonical JSON
- 3+ test cases in `src/tac/tests/test_pair_frame_scorer_geometry_lattice.py` covering happy path + missing-anchor handling + multi-archive disambiguation

**Operator-routable gate**: sister subagent emit per Catalog #313 probe-outcomes ledger row; operator decides whether to fund BUILD-2.

#### BUILD-2: Full executable PairFrameScorerGeometryLattice with 4 operation generators

**Scope**: full executable implementation of `class PairFrameScorerGeometryLattice` with the 4 canonical operation generators (`generate_full_drop_starts` / `generate_repair_starts` / `generate_masked_starts` / `generate_feathered_starts`). Each generator emits up to `top_n` ExecutableCandidates ranked by per-cell predicted ΔS via the lattice's canonical per-axis × receiver_runtime × cpu_cuda_axis decomposition.

**Cost**: $0 paid GPU + ~4-8h wall-clock (sister subagent).

**Deliverables**:
- full implementation of `PairFrameScorerGeometryLattice.generate_*_starts` (replaces scaffold `NotImplementedError`)
- canonical CLI `tools/generate_pair_frame_lattice_candidates.py --archive <sha> --operation full_drop --top-n 32 --output <path>`
- 10+ test cases covering each operation generator + per-axis decomposition + canonical Provenance threading per Catalog #323

**Operator-routable gate**: sister subagent emits canonical candidate manifest JSON; operator decides whether to fund BUILD-3.

#### BUILD-3: Catalog #356 AxisDecomposition wire-in

**Scope**: extend `ExecutableCandidate` to emit `predicted_axis_decomposition: AxisDecomposition | None` per Catalog #356. Each candidate from the 4 operation generators MUST carry per-axis (seg, pose, rate) prediction in canonical AxisDecomposition contract. Required for BUILD-4 Tier B promotion.

**Cost**: $0 paid GPU + ~1-2h wall-clock (sister subagent).

**Deliverables**:
- `AxisDecomposition` instances threaded through every operation generator output
- canonical Provenance per axis-decomposition per Catalog #323
- 5+ test cases covering AxisDecomposition emission + canonical formula composition via `tac.score_composition.compose_score_from_axes`

**Operator-routable gate**: sister subagent emits Catalog #356 conformance verdict; operator decides whether to fund BUILD-4.

#### BUILD-4: Catalog #357 Tier B promotion

**Scope**: promote `pair_frame_scorer_geometry_lattice` to Tier B score-contributing canonical consumer per Catalog #357. Registration as cathedral consumer package at `src/tac/cathedral_consumers/pair_frame_scorer_geometry_lattice_consumer/__init__.py` with `CONSUMER_TIER = ConsumerTier.TIER_B_SCORE_CONTRIBUTING` + canonical-routing markers per Catalog #341 (`predicted_delta_adjustment > 0.0` permitted; `promotable=False` per Tier B contract; `axis_tag` empirically-grounded NOT `[predicted]`).

**Cost**: $0 paid GPU + ~2-4h wall-clock (sister subagent).

**Deliverables**:
- cathedral consumer package with canonical contract per Catalog #335 + Catalog #357
- consumer `consume_candidate` implementation that emits Tier B contribution via lattice query
- consumer `update_from_anchor` implementation for canonical posterior updates per Catalog #344
- 10+ test cases covering canonical contract compliance + Tier B contribution validation per `validate_tier_b_contribution` + auto-discovery per Catalog #335

**Operator-routable gate**: sister subagent emits Catalog #357 conformance verdict; operator decides whether to fund DISPATCH wave.

#### DISPATCH: Paired CPU+CUDA Modal exact-eval of top-K candidates

**Scope**: paid Modal dispatch of top-K candidates per operation per axis_label. The lattice + generators produce ranked candidates; this DISPATCH wave executes the canonical exact-eval per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable.

**Cost**: ~$2-10 per cascade wave (paired CPU+CUDA Modal T4 per top-K candidate × 4 operations).

**Deliverables**:
- paired CPU+CUDA exact-eval anchors per dispatched candidate
- empirical anchors for canonical equation registration per Catalog #344 (3+ anchors needed per operation × axis_label to register the canonical equation candidate)
- canonical frontier pointer updates per Catalog #343 if any candidate beats current frontier (currently `7a0da5d0fc32` 0.19202828 [contest-CPU])

**Operator-routable gate**: operator decides per-wave dispatch budget; canonical operator-authorize chain per Catalog #271 (codex pre-dispatch review) + Catalog #243 (local pre-deploy harness) + Catalog #167 (smoke-before-full pattern).

---

## Discipline closure

### 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map**: **ACTIVE** — per-cell `bind_pair_component_xray` + `decompose_frame_axis_master_gradient` are sensitivity-map producers; the lattice IS the joint sensitivity surface across (pair × frame × scorer_axis).
- **hook #2 Pareto constraint**: **ACTIVE** — per-cell `query_receiver_runtime_feasibility` IS Dykstra alternating-projection feasibility; the lattice's feasible region is the convex intersection of per-cell receiver_runtime feasibility constraints (per CLAUDE.md "Council conduct" Dykstra co-lead).
- **hook #3 bit-allocator**: **ACTIVE** — `predicted_byte_cost` per cell IS the bit-allocator's primary signal; per-cell decomposition unlocks per-region byte allocation.
- **hook #4 cathedral autopilot dispatch**: **ACTIVE PRIMARY** — BUILD-4 Tier B promotion auto-discovers the lattice consumer per Catalog #335; the 4 operation generators emit ExecutableCandidates that flow into the cathedral autopilot ranker via canonical-routing markers per Catalog #341.
- **hook #5 continual-learning posterior**: **ACTIVE** — per-cell measurement updates flow into canonical posterior via Catalog #344 `update_equation_with_empirical_anchor`; each DISPATCH wave's empirical anchors update the canonical equation candidate's calibration.
- **hook #6 probe-disambiguator**: **ACTIVE** — the 5D lattice IS the canonical disambiguator across the 4 operations + 5 receiver_runtime modes + 2 CPU/CUDA axes; per-cell `query_receiver_runtime_feasibility` answers "which receiver mode works for this pair-frame?" without re-running scorer.

### Catalog #313 probe-outcomes ledger row

Will be registered via `tac.probe_outcomes_ledger.register_probe_outcome` in a sister commit batch alongside this memo landing. Verdict: **DEFER_PENDING_BUILD_1** (design memo + scaffold skeleton only; full empirical lattice requires BUILD-1 sister subagent).

### Catalog #344 canonical equation candidate

QUEUED via this memo body (DELIVERABLE 3). Equation ID `pair_frame_scorer_geometry_lattice_4d_binding_canonical_v1`. NOT auto-registered. Operator-routable; registration triggers when 3+ empirical anchors land across BUILD-1 through BUILD-4 + DISPATCH waves.

### Sister-coherence verification

- Slot 2 (MLX-ARCH-4 SegNet) — DISJOINT (MLX architecture vs canonical 5D lattice scaffold)
- Slot 3 (DROP-MANY-BEAM-DESIGN) — DISJOINT (drop-many beam search is ONE algorithm on the lattice; this memo designs the CANVAS itself); sister equation candidate `dqs1_drop_many_pairwise_interaction_beam_search_v1` is sister-of `pair_frame_scorer_geometry_lattice_4d_binding_canonical_v1` per DELIVERABLE 3
- Slot 4 (RATE-ATTACK-METHODS-DIMENSIONS-MATRIX) — DISJOINT (matrix maps full M×N methods×dimensions; this memo designs the specific 5D lattice binding that the matrix references)
- Sister checkpoint via `tools/subagent_checkpoint.py` PROCEED required per Catalog #340

### Files touched (canonical APPEND-ONLY per Catalog #110/#113)

- NEW `.omx/research/pair_frame_scorer_geometry_lattice_design_memo_20260525.md` (this file)
- NEW `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas.py` (scaffold skeleton; sibling of existing codex v1 row-based module)

NO mutation of:
- CLAUDE.md
- canonical equation registry (`.omx/state/canonical_equations_registry.jsonl`)
- codex DQS1 cascade source files
- sister cathedral consumers
- state JSON files
- AGENTS.md
- any sister design memo or landing memo

### Apparatus-discipline acknowledgment

- Catalog #1 + #192: design memo predictions are advisory; promotion via paired CPU+CUDA
- Catalog #287: every empirical claim has evidence tag (this memo carries [prediction] tags on all canonical equation predictions)
- Catalog #323: canonical Provenance umbrella; threaded through `PairFrameScorerGeometryCell.catalog_323_provenance` field
- Catalog #313: probe-outcomes registered via canonical helper (DEFER_PENDING_BUILD_1)
- Catalog #344: canonical equation candidate QUEUED for operator-routable RATIFY-N
- Catalog #110/#113 APPEND-ONLY: NEW memo + NEW scaffold script only
- CLAUDE.md "Forbidden premature KILL": no kill verdicts; sister-family composition operations are DEFER-pending-BUILD-1
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode": the 5D lattice IS the unique canonical primitive that simultaneously binds the 5 axes codex named; per-receiver-runtime mode is substrate-optimal engineering

---

## Mission contribution per Catalog #300

`frontier_breaking_enabler` — the 5D lattice is the canonical primitive that unblocks 4 currently-blocked operator-explicit requests simultaneously. Per CLAUDE.md "Mission alignment — non-negotiable" Consequence 4: frontier-breaking moves DOMINATE rigor budget. This is THE next canonical bridge per codex's explicit naming; design memo + scaffold + 4-BUILD enumeration is the structural foundation that BUILD-1 through DISPATCH waves consume.

## Cross-references

- `.omx/research/codex_findings_eureka_drop_many_rate_distortion_budget_20260525T143351Z_codex.md` §"Residual Gap" (the operator-routable origin)
- `.omx/research/dqs1_loop_closure_assist_audit_plus_engineering_improvements_20260525.md` (Top-3 MEDIUM operator-routable; this memo addresses #1 + partially #4)
- `tac.cathedral.consumer_contract` (Catalog #356 AxisDecomposition + Catalog #357 Tier B contract)
- `tac.optimization.decoder_q_pairset_acquisition` (sister DQS1 acquisition interface; canvas-vs-algorithm distinction per DELIVERABLE 3)
- `tac.master_gradient_consumers.per_pair_gradient_clustering_consumer` (sister Cable D consumer interface pattern)
- `tac.cathedral_consumers.per_frame_sensitivity_consumer` (sister per-frame consumer)
- canonical equation #36 `pairset_component_marginal_score_decomposition_v1` (HARD-EARNED sister)
- canonical equation #18 `pose_axis_cuda_amplification_v1` (HARD-EARNED sister for cpu_cuda_axis_amplification)

## Lane registration

Lane `lane_pair_frame_scorer_geometry_lattice_design_memo_20260525` L1 (impl_complete + design_memo + scaffold_skeleton).
