# Per-trainer wire-in → AUTOMATIC directive (operator pivot 2026-05-14)

**Lane**: `lane_per_trainer_optimization_wire_in_20260514`
**Active subagent**: `aedc327d17f37d53b` (PER-TRAINER-WIRE-IN)
**Operator directive (verbatim 2026-05-14)**: *"make it automatic"*
**Tag**: `research_only=true`; canonical persistence handoff per CLAUDE.md "Subagent coherence-by-default" mandatory `.omx/research/*_directive_*` pre-read.

## Spec pivot

REPLACE manual per-trainer 5-LOC backport across 20+ trainers with AUTOMATIC three-layer approach. The operator doesn't want one-time backports — they want the wire-in to be STRUCTURALLY AUTOMATIC for all future + all existing trainers.

## Three automatic layers (build all three)

### Layer 1 — Canonical trainer skeleton with helpers BAKED IN

Update `src/tac/substrates/_shared/trainer_skeleton.py` (the canonical trainer base) to inherit the wire-in pattern automatically:

```python
# Add to trainer_skeleton helpers:

def build_optimized_training_context(
    args: argparse.Namespace,
    scorers,
    gt_pairs: torch.Tensor,
    substrate_model: torch.nn.Module,
    device: torch.device,
) -> "OptimizedTrainingContext":
    """Canonical Tier-1 optimization context — automatic for all trainers.

    Returns a context object with:
      - gt_cache (GTScorerCache if --enable-gt-scorer-cache)
      - autocast_ctx (AutocastConfig honoring --enable-autocast-fp16)
      - substrate_model (wrapped via compile_with_fallback if --enable-torch-compile)
      - score_pair_components_fn (cache-aware variant if cache enabled)
    """
    from tac.training_optimization import (
        AutocastConfig, CompileConfig, build_gt_scorer_cache, compile_with_fallback,
    )
    from tac.substrates.score_aware_common import (
        score_pair_components, score_pair_components_with_cache,
    )

    gt_cache = (
        build_gt_scorer_cache(scorers=scorers, gt_pairs=gt_pairs, device=device)
        if getattr(args, "enable_gt_scorer_cache", True)
        else None
    )

    if getattr(args, "enable_torch_compile", False):
        substrate_model = compile_with_fallback(
            substrate_model,
            CompileConfig(enabled=True, mode="default", fallback_on_error=True),
        )

    score_fn = (
        functools.partial(score_pair_components_with_cache, gt_cache=gt_cache, scorers=scorers)
        if gt_cache is not None
        else functools.partial(score_pair_components, gt_pairs=gt_pairs, scorers=scorers)
    )

    autocast_cfg = AutocastConfig(
        enabled=getattr(args, "enable_autocast_fp16", False),
        dtype=torch.float16,
        device_type=device.type,
    )

    return OptimizedTrainingContext(
        gt_cache=gt_cache,
        substrate_model=substrate_model,
        score_fn=score_fn,
        autocast_cfg=autocast_cfg,
    )
```

Plus the canonical TIER_1 flag declarations get added to a SHARED manifest source:

```python
# Canonical optimization flags that every substrate trainer inherits via
# trainer_skeleton.merge_optimization_flags():

OPTIMIZATION_FLAGS_MANIFEST = {
    "--enable-autocast-fp16": {...},
    "--enable-torch-compile": {...},
    "--enable-gt-scorer-cache": {"default": True, ...},
}

def merge_optimization_flags(trainer_tier_1_manifest: dict) -> dict:
    """Merge canonical optimization flags into trainer's TIER_1 manifest."""
    return {**OPTIMIZATION_FLAGS_MANIFEST, **trainer_tier_1_manifest}
```

Future substrate trainers just do `TIER_1_OPERATOR_REQUIRED_FLAGS = merge_optimization_flags({...substrate-specific flags...})` and they get the canonical optimization flags for free.

### Layer 2 — Automatic AST patcher tool

Build `tools/auto_wire_in_training_optimization.py`:

```python
"""Automatic AST-based wire-in of canonical optimization helpers into substrate trainers.

For each experiments/train_substrate_*.py:
  1. Parse via ast.parse()
  2. Detect MISSING imports → inject
  3. Detect MISSING TIER_1 manifest entries → inject via merge_optimization_flags()
  4. Detect MISSING build_optimized_training_context call → inject after scorers_loaded
  5. Detect MANUAL autocast/compile wraps → consolidate to canonical helper calls
  6. Write back with --expected-content-sha256 commit
"""

def auto_wire_in_trainer(trainer_path: Path) -> WireInResult:
    """AST-based automatic backport. Idempotent."""
    ...

def auto_wire_in_all(scan_root: Path = Path("experiments")) -> list[WireInResult]:
    """Backport all substrate trainers automatically.

    Skip exclusion list (sister-subagent ownership):
      - train_substrate_c6_e4_mdl_ibps.py
      - train_substrate_d1_segnet_margin_polytope.py
      - train_substrate_d4_wyner_ziv_frame_0.py
    """
    ...
```

This tool can be re-run idempotently — if a trainer already has the canonical helpers, it's a no-op. Future trainers get auto-patched in CI if a hook is wired in.

Idempotency contract: the tool MUST be safe to run repeatedly without changing already-patched trainers. AST-level detection (not text-level) for the canonical-helper imports + function calls.

### Layer 3 — STRICT preflight gate refusing trainers without wire-in

Claim a new Catalog # via:
```bash
.venv/bin/python tools/claim_catalog_number.py claim --commit-via-serializer --reason "Substrate trainers must use canonical tac.training_optimization helpers (automatic wire-in enforcement)"
```

Land `check_substrate_trainers_use_canonical_optimization_helpers` STRICT preflight gate:

- Scans `experiments/train_substrate_*.py`
- For each trainer, requires either:
  1. Imports `from tac.substrates._shared.trainer_skeleton import build_optimized_training_context` AND calls it OR
  2. Imports + uses `tac.training_optimization.GTScorerCache` / `autocast_aware_forward` / `compile_with_fallback` directly (manual but acceptable equivalent) OR
  3. Carries same-line `# OPTIMIZATION_HELPERS_WAIVED:<reason>` waiver
- Sister-subagent ownership exemptions (C6/D1/D4) via path-prefix waiver

Wire-in strict-from-byte-one after the AST patcher drives live count to 0.

## Test plan

For each layer:
- Layer 1 (skeleton): 20+ tests in `test_trainer_skeleton_optimized_context.py` covering helper composition, cache-aware vs uncached path equivalence, AutocastConfig honoring args, compile fallback semantics
- Layer 2 (patcher): 25+ tests in `test_auto_wire_in_training_optimization.py` covering AST detection, idempotency, exclusion list, write-back-with-sha
- Layer 3 (gate): 15+ tests in `test_check_NNN_canonical_optimization_helpers.py` covering positive (no canonical helper → refused), negative (canonical helper present → pass), waiver semantics, sister-subagent exemption

Plus integration test: run the AST patcher on a fresh fixture trainer; verify it's patched correctly; re-run; verify idempotent.

## Commit batching

- Batch 1: Layer 1 (canonical skeleton extension) + 20+ tests
- Batch 2: Layer 2 (AST patcher tool) + 25+ tests + CLI surface
- Batch 3: Run Layer 2 across all ~25 in-scope trainers; commits per-trainer via canonical serializer (Catalog #157+#216 protected)
- Batch 4: Layer 3 (STRICT preflight gate) + 15+ tests + strict-flip per atomicity rule
- Batch 5: Local smoke verification across patched trainers; document any blockers

## Validation per Layer 2 patched trainer

After AST patcher modifies a trainer:
1. `pytest <trainer's test file>` — must pass
2. `.venv/bin/python <trainer_path> --smoke --epochs 10 --device cpu --advisory-cpu-explicitly-waived` — must pass (no regression)
3. Catalog #172/#179/#180 preflight gates remain at strict @ 0
4. If a trainer's smoke FAILS post-patch, AST patcher reverts that trainer's edit and documents the blocker

## Memory + reporting

`~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_per_trainer_wire_in_automatic_landed_20260514.md` documenting:
- 3-layer architecture
- All trainers patched (count + list)
- Catalog # claimed + STRICT-flip verification
- Smoke test results per trainer
- 5 operator-routable decisions (e.g., first Modal A100 smoke to validate empirical speedup; canonical autocast default policy; CI hook to auto-run AST patcher on every PR)

## Cross-refs

- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_tier_1_optimization_batch_landed_20260514.md` (helpers landed)
- `src/tac/training_optimization/` (the canonical helpers)
- `src/tac/substrates/_shared/trainer_skeleton.py` (the canonical skeleton; extend it)
- CLAUDE.md "Subagent coherence-by-default" non-negotiable (this directive is the canonical persistence-based handoff)
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" (Layer 3 STRICT gate IS the self-protection)

Tagged `research_only=true`. NO score claims. NO GPU spend by this directive. The in-flight subagent picks up this layered spec on next checkpoint cycle or completion handoff.
