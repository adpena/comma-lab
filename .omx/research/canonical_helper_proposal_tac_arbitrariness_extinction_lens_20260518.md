# Canonical Helper Proposal — `tac.arbitrariness_extinction_lens`

**Subagent**: `lane_arbitrariness_extinction_meta_lens_systematic_audit_20260518`
**Status**: PROPOSAL (DO NOT BUILD) — awaits operator decision
**Captured at UTC**: `2026-05-18T21:18:00Z`
**Source**: operator standing directive 2026-05-18 (5-path resolution: experimental / analytical_solve / formula / learned / self_alien_tech)

## Purpose

After today's systematic audit (52 rows; total predicted ΔS envelope [-0.139, -0.026]; ~$120 total cost; 31 rows at $0 cost), there is a clear META-pattern: **the apparatus accumulates arbitrary defaults silently**. New substrate trainers inherit `lr=5e-4`, `ema_decay=0.997`, `weight_decay=1e-5`, `batch_size=4|8|16|32` (VRAM-driven), `K=64`, `sigma=15`, `threshold=0.5` etc. without explicit per-substrate-design justification.

A canonical helper that AUTOMATES this audit recurringly would:

1. Surface NEW arbitrary defaults the moment they land
2. Provide per-value 5-path classification heuristics
3. Integrate with cathedral autopilot ranker as fresh-research-probe source
4. Compose with Catalog #303 (cargo-cult audit) + Catalog #229 (premise verification) + HNeRV parity L6 (score-domain Lagrangian)

## Proposed module: `src/tac/arbitrariness_extinction_lens/`

### Files

```
src/tac/arbitrariness_extinction_lens/
  __init__.py
  scanner.py        # AST-based scanner for hardcoded numerical/categorical defaults
  classifier.py     # per-value 5-path heuristic classifier
  contract.py       # ArbitraryValue + ResolutionPath + ResolutionVerdict dataclasses
  ranker.py         # rank-by-EV/$ + cluster-by-resolution-path
  cathedral_wirein.py  # autopilot ranker integration
  composition.py    # composes with Catalog #303 + #229 + HNeRV-L6
```

### Public API

```python
# Contract
@dataclass(frozen=True)
class ArbitraryValue:
    value_id: str
    file_path: str
    current_value: str | int | float
    is_arbitrary: Literal[True, False, "hard-earned-empirical", "contest-fixed"]
    detection_source: str  # AST signature

@dataclass(frozen=True)
class ResolutionVerdict:
    recommended_path: Literal[
        "experimental", "analytical_solve", "formula",
        "learned", "self_alien_tech", "composition", "contest_fixed"
    ]
    predicted_replacement: str
    predicted_ev_delta_s: tuple[float, float]
    cost_envelope_usd: float
    cheaper_alternative_path: str
    blocking_dependencies: list[str]
    literature_citation: str
    canonical_helper_repo_link: str
    rationale: str

# Scanner
def scan_arbitrary_defaults(
    *,
    scan_roots: Sequence[Path],
    file_globs: Sequence[str] = ("**/*.py",),
    exempt_path_markers: Sequence[str] = ("experiments/results/", "_intake_", "/tests/"),
) -> list[ArbitraryValue]:
    """AST-walk for hardcoded numerical/categorical defaults in argparse + Config dataclasses."""

# Classifier (heuristic)
def classify_resolution_path(value: ArbitraryValue) -> ResolutionVerdict:
    """Per-value 5-path classifier with literature-citation lookup."""

# Ranker
def rank_arbitrary_values(
    values: Sequence[ArbitraryValue],
    verdicts: Mapping[str, ResolutionVerdict],
) -> list[tuple[ArbitraryValue, ResolutionVerdict, float]]:
    """Sort by |EV.lower| / cost_usd descending (with cost=0 → 1000× boost)."""

# Cathedral autopilot wire-in (hook #4 per Catalog #125)
def emit_fresh_research_probes_to_autopilot(
    ranked: list[tuple[ArbitraryValue, ResolutionVerdict, float]],
    autopilot_queue_path: Path = Path(".omx/state/autopilot_candidate_queue.jsonl"),
) -> None:
    """Append top-K arbitrary-value rows as research-probe candidates to autopilot queue."""
```

### Detection patterns (scanner heuristics)

```python
# Argparse default detector
parser.add_argument("--lr", type=float, default=5e-4)  # DETECT: literal float in default=
parser.add_argument("--ema-decay", type=float, default=0.997)  # DETECT
parser.add_argument("--batch-size", type=int, default=8)  # DETECT (int literal)

# Constant detector
LEGACY_BROTLI_QUALITY = 10  # DETECT: module-level constant
DEFAULT_COS_KEEP_THRESHOLD = 0.30  # DETECT
QINT_LEVELS = (1, 3, 7, 15, 31)  # DETECT: tuple-of-int hardcoded grid

# Dataclass field default detector
@dataclass
class Config:
    codebook_size: int = 64  # DETECT
    sigma: float = 15.0  # DETECT
```

### Exempt patterns (NOT arbitrary)

```python
# Same-line waiver
sigma = 15.0  # ARBITRARINESS_EXTINCTION_OK:Selfcomp-verified-empirical-anchor
parser.add_argument("--lr", default=5e-4)  # ARBITRARINESS_EXTINCTION_OK:Quantizr-paradigm-canonical

# Contest-fixed values
SCORE_RATE_DENOM_BYTES = 37_545_489  # CONTEST_FIXED
sqrt_10_pose_weight = math.sqrt(10)  # CONTEST_FIXED

# Hard-earned-empirical (with citation)
EMA_DECAY_QUANTIZR_EMPIRICAL = 0.997  # HARD_EARNED_EMPIRICAL:Quantizr-0.33-archive
```

### Composition with sister gates

- **Catalog #303** (`check_substrate_design_memo_has_cargo_cult_audit_section`): per-substrate hard-earned-vs-cargo-culted classification at the design-memo surface. Arbitrariness-lens automates the AST scan at the source-text surface.
- **Catalog #229** (`check_subagent_landing_includes_premise_verification_evidence`): premise-verification-before-edit pattern. Arbitrariness-lens supplements with PRE-substrate-scaffold-landing scan: every NEW substrate trainer's argparse defaults must pass through the lens before commit.
- **CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable**: every λ choice in the action principle MUST be either canonical-derivation or empirically-anchored. The lens forces explicit citation per value.
- **Catalog #125** (`check_subagent_landing_has_solver_wire_in`): the lens's `cathedral_wirein.emit_fresh_research_probes_to_autopilot` IS hook #4 wire-in for the audit output.

### Sister to existing helpers

- `src/tac/non_arbitrariness.py` (lane non-arbitrariness from earlier; META: itself has DEFAULT_COS_KEEP=0.30 and DEFAULT_COS_PRUNE=0.85 hardcoded — would be flagged by its own sister lens)
- `tac.preflight_rudin_daubechies.PreflightSLIMRiskScorer` (interpretability gate)

## Proposed CLI

```bash
# Operator-runnable scan
.venv/bin/python tools/audit_arbitrariness_extinction.py \
    --scan-root src/tac \
    --scan-root experiments \
    --top-k 20 \
    --output .omx/state/arbitrariness_extinction_audit_<utc>.jsonl

# Specific resolution-path filter
.venv/bin/python tools/audit_arbitrariness_extinction.py \
    --filter-resolution-path formula \
    --max-cost-usd 0
```

## Proposed STRICT preflight gate (operator decision required)

**Catalog #~333** `check_substrate_design_memo_has_arbitrariness_audit_section`:
- Refuses substrate design memos at `.omx/research/*_design_<YYYYMMDD>.md` dated >= 2026-05-18 lacking the literal section header `## Arbitrariness audit per layer`
- Sister to Catalog #290 / #303 / #294 / #305 / #296 (design-memo discipline gates)
- Initial wire-in: WARN-ONLY per CLAUDE.md "Strict-flip atomicity rule"
- Strict-flip pending operator-routed backfill of in-flight design memos

## Operator decision required

1. **Greenlight build of `tac.arbitrariness_extinction_lens` module?** (Estimated build: ~600 LOC, ~30 tests, 1 subagent)
2. **Greenlight Catalog #~333 STRICT preflight gate proposal?** (Sister to Catalog #303; would land WARN-ONLY then strict-flip)
3. **Greenlight CLAUDE.md amendment proposal?** (See sister memo `claude_md_non_negotiable_proposal_arbitrariness_extinction_meta_lens_20260518.md`)
4. **Greenlight cathedral autopilot wire-in?** (Hook #4 per Catalog #125)

## Cross-references

- `.omx/state/arbitrariness_extinction_audit_20260518.jsonl` — the canonical first audit (52 rows; this proposal would automate)
- `.omx/research/arbitrariness_extinction_audit_top_50_ranked_20260518.md` — operator-readable ranked summary
- Top-10 routing directives at `.omx/research/codex_routing_directive_arbitrariness_extinction_top*_*_20260518.md`
- Memory entry `feedback_arbitrariness_extinction_meta_lens_systematic_audit_landed_20260518.md`

## Cost envelope

- Audit lens build: ~600 LOC + tests, $0 GPU
- Catalog #333 gate: ~50 LOC + tests, $0
- CLAUDE.md amendment: 1 commit
- Cathedral autopilot wire-in: ~20 LOC, $0
- Total: $0 GPU, ~1.5h subagent wall-clock
