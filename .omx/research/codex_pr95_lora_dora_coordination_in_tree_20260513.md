# Codex in-tree work coordination for pr95_lora_dora (2026-05-13)

**Lane**: `lane_codex_math_correction_and_coordination_20260513` (L0 SKETCH; companion to score-math audit memo).
**Operator directive 2026-05-13**: "Do NOT re-edit those files; codex owns the cleanup" + "produce a coordination memo describing the intended API contract so codex can land it independently".

## Codex's in-tree state (verified)

As of the most recent commit `cd40d859` ("pr95: add LoRA adapter break-even budget math"):

```
src/tac/substrates/pr95_lora_dora/budget.py        207 LOC  (NEW — codex)
src/tac/substrates/pr95_lora_dora/tests/test_budget.py  63 LOC  (NEW — codex)
.omx/research/stack_of_stacks_dispatch_blockers_20260513_codex.md  39 LOC (NEW — codex review)
```

Earlier in the same session at commit `77b36949`:
```
src/tac/substrates/pr95_lora_dora/__init__.py            74 LOC
src/tac/substrates/pr95_lora_dora/architecture.py       336 LOC
src/tac/substrates/pr95_lora_dora/archive.py            239 LOC
src/tac/substrates/pr95_lora_dora/inflate.py            199 LOC
src/tac/substrates/pr95_lora_dora/pr95_base.py           81 LOC
src/tac/substrates/pr95_lora_dora/score_aware_loss.py    46 LOC
src/tac/substrates/pr95_lora_dora/tests/test_architecture.py  271 LOC
src/tac/substrates/pr95_lora_dora/tests/test_archive.py       194 LOC
src/tac/substrates/pr95_lora_dora/tests/test_inflate.py       127 LOC
```

The prompt mentioned codex "applied 16 ruff auto-fixes" + "fixed architecture.py SIM108 (ternary) at line 289-290" + "fixed pr95_base.py zip(strict=True) at line 71". I verified the SIM108 + strict=True locations (architecture.py:289-290 and pr95_base.py:71 region) and the budget helper landing.

**Coordination decision per operator directive**: this Claude session will NOT modify any file under `src/tac/substrates/pr95_lora_dora/`. Codex owns that surface during this session.

## API contract for the byte-break-even helper (already implemented by codex)

Codex's canonical helper module `src/tac/substrates/pr95_lora_dora/budget.py` exposes:

### Module constants

- `DEFAULT_TIER_C_LAYER_DIMS: tuple[tuple[str, int, int], ...]` — 6 PR95 upsample conv blocks (name, out_dim, in_dim_flattened) per the deconstruction memo's Tier C inventory.

### Dataclasses

```python
@dataclass(frozen=True)
class AdapterLayerBudget:
    name: str
    out_dim: int
    in_dim: int
    rank: int
    kind: AdapterKind         # Literal["lora", "dora"]
    trainable_params: int
    raw_trailer_bytes: int

@dataclass(frozen=True)
class AdapterBreakEven:
    raw_trailer_bytes: int
    rate_score_penalty: float       # = 25 * bytes / 37_545_489
    required_seg_reduction: float   # = penalty / 100  (linear seg axis)
    pose_operating_point: float     # default 3.4e-5
    required_pose_reduction_exact: float  # exact sqrt inverse
    pose_only_feasible: bool        # False if penalty exceeds current pose term
    residual_score_after_zero_pose: float
    evidence_grade: str = "[prediction; closed-form adapter break-even]"
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
```

### Pure functions (no torch, no archive load, no scorer)

```python
def adapter_trainable_params(*, out_dim, in_dim, rank, kind) -> int
def adapter_raw_trailer_bytes(*, name, out_dim, in_dim, rank, kind) -> int
def tier_c_layer_budgets(*, rank=8, kind="lora") -> tuple[AdapterLayerBudget, ...]
def tier_c_trainable_params(*, rank=8, kind="lora") -> int
def tier_c_raw_trailer_bytes(*, rank=8, kind="lora") -> int

def rate_score_penalty_for_bytes(byte_delta: int) -> float

def exact_pose_reduction_for_score_delta(
    pose_dist: float,
    score_delta: float,
) -> tuple[float, bool, float]:
    """
    Returns (required_pose_reduction, pose_only_feasible, residual_after_zero_pose).
    Uses exact sqrt inversion: solves
        sqrt(10*(p - Δp)) = sqrt(10*p) - ΔS  for Δp
    which gives
        Δp = ΔS * (2*sqrt(10*p) - ΔS) / 10
    feasibility check: if ΔS >= sqrt(10*p), pose alone cannot offset; returns
    (p, False, residual = ΔS - sqrt(10*p)).
    """

def adapter_break_even(
    *,
    raw_trailer_bytes: int,
    pose_operating_point: float = 3.4e-5,
) -> AdapterBreakEven
```

### Test invariants (verified by codex's `test_budget.py`)

1. `tier_c_trainable_params(rank=8, kind="lora") == 17_416` (matches deconstruction memo §"Adapter parameter budget").
2. `tier_c_trainable_params(rank=8, kind="dora") == 17_416 + 620` (one extra magnitude per output channel).
3. `tier_c_raw_trailer_bytes(rank=8, kind="lora") == len(encode_lora_trailer(records))` — math-as-code MATCHES the archive encoder. This is the canonical bidirectional check.
4. `tier_c_raw_trailer_bytes(rank=8, kind="dora") == len(encode_lora_trailer(records_with_magnitudes))` — same bidirectional check for DoRA.
5. `adapter_break_even(raw_trailer_bytes=21_000)` returns:
   - `0.0139 < rate_score_penalty < 0.0141` ✓ (matches 21000·6.66e-7 = 0.01398)
   - `1.39e-4 < required_seg_reduction < 1.41e-4` ✓ (matches 0.01398/100 linear)
   - `3.1e-5 < required_pose_reduction_exact < 3.3e-5` ✓ (exact sqrt inverse; the OLDER linearized estimate of ~5e-7 fails this assertion by ~50x)
   - `required_pose_reduction_exact > 50 * 5e-7` ✓ (explicit anti-regression guard against the old linearization)
   - `pose_only_feasible is True` ✓ (barely; ΔS = 0.01398 vs current pose term 0.01844)
   - `score_claim is False`, `ready_for_exact_eval_dispatch is False` ✓ (prediction grade)
6. `adapter_break_even(raw_trailer_bytes=40_000)` returns `pose_only_feasible is False` and `residual_score_after_zero_pose > 0.0` ✓ (40 KB penalty 0.0266 exceeds 0.01844 pose term; pose alone cannot offset).

### Integration contract for trainer / dispatcher

The trainer (`experiments/train_substrate_pr95_lora_dora.py` — not yet built; Phase 4 deliverable per LoRA/DoRA landed memo §5) is expected to:

1. Import `from tac.substrates.pr95_lora_dora.budget import adapter_break_even, tier_c_raw_trailer_bytes`.
2. Compute the predicted trailer bytes from the adapter rank/kind config at INIT time.
3. Compute `adapter_break_even(raw_trailer_bytes=predicted_bytes, pose_operating_point=current_pose_avg_estimate)`.
4. Log the break-even verdict to the manifest BEFORE GPU dispatch starts.
5. If `pose_only_feasible is False`, **refuse to declare `score_claim_band`** without explicit composition with seg-axis reduction.

The dispatcher / operator-authorize wrapper (also Phase 4) is expected to:

1. Read the trainer's break-even manifest.
2. Refuse dispatch if `cost_band est_cost_usd > 5.0` AND `pose_only_feasible is False` AND no seg-composition declared.

These are MATH-AS-CODE invariants. They prevent the LoRA/DoRA dispatch from being routed on the bad linearization that originally claimed "1e-5 pose reduction → -0.0027 score" tractability.

## Files Claude WILL edit (this session)

- `.omx/research/codex_pr95_lora_dora_coordination_in_tree_20260513.md` (this memo, NEW)
- `.omx/research/score_math_rigor_audit_post_codex_correction_20260513.md` (companion audit memo, NEW)
- `.omx/research/pr95_artifact_deconstruction_20260513.md` (already corrected by codex; no edit needed this session)
- `.omx/research/council_a1_pr95_pr98_deliberation_20260512.md` (math-correction edit at line 62; sister bug to LoRA/DoRA)
- `.omx/research/grand_council_pose_axis_non_hnerv_a1_plus_lapose_20260513.md` (math-correction edit at line 22; sister bug)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_pr95_artifact_lora_dora_surgery_landed_20260513.md` (math-correction edit at line 86; the original bad line)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_codex_math_correction_pr95_lora_dora_landed_20260513.md` (this session's landing memo, NEW)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/MEMORY.md` (top-line index update, NEW entry only)

## Files Claude will NOT edit (codex owns)

- `src/tac/substrates/pr95_lora_dora/*` (all files)
- `src/tac/substrates/pr95_lora_dora/tests/*` (all files)

## No conflict with sister subagents

Sister subagent `lane_online_research_bleeding_edge_synthesis_20260513` is doing online research (independent surface). No file-overlap.

## Wire-in hooks per Catalog #125

All 6 hooks N/A this landing (pure coordination + memo edits). Declared explicitly to honor the "no silent omission" rule.

## References

- Codex commit: `cd40d859` (the math-as-code helper landing)
- Codex prior commit: `77b36949` (substrate package landing)
- Codex review memo: `.omx/research/stack_of_stacks_dispatch_blockers_20260513_codex.md`
- Codex coordination synthesis: `.omx/research/codex_coordination_shared_page_synthesis_20260513.md`
- Codex frontier roadmap: `.omx/research/sub017_frontier_innovation_roadmap_20260513_codex.md`
- Score-math rigor audit: `.omx/research/score_math_rigor_audit_post_codex_correction_20260513.md`
- LoRA/DoRA landed memo (math-corrected line 86): `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_pr95_artifact_lora_dora_surgery_landed_20260513.md`
- Deconstruction memo (math-corrected lines 194-209 by codex): `.omx/research/pr95_artifact_deconstruction_20260513.md`
- Canonical helper: `src/tac/substrates/pr95_lora_dora/budget.py`
- Canonical tests: `src/tac/substrates/pr95_lora_dora/tests/test_budget.py`
