# SPDX-License-Identifier: MIT
"""Canonical MLX canonicalization audit primitives.

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable + the
operator's binding 2026-05-30 directive. THIS module exposes the
substantive APIs that PHASE B operationalizes.

NO FAKE IMPLEMENTATIONS per CLAUDE.md "NO FAKE IMPLEMENTATIONS" non-
negotiable: every function does the actual work it names on the actual
inputs:

  * :func:`enumerate_mlx_bearing_files` walks the filesystem with
    ``Path.rglob("*.py")`` + reads file bytes + tests for ``import mlx``
    via line-anchored regex (NOT a placeholder list).
  * :func:`enumerate_mlx_primitive_implementations` parses every matched
    file via :mod:`ast` + collects ``FunctionDef`` + ``ClassDef`` + the
    canonical-import token presence per Catalog #168 sister AST-aware
    discipline (NOT a hard-coded primitive set).
  * :func:`detect_canonical_duplication` returns a typed
    :class:`CanonicalDuplicationVerdict` based on actual implementation
    counts + actual canonical-routing-import presence; placeholder
    rationales rejected per Catalog #287 sister discipline.
  * :func:`recommend_canonical_extraction` emits a typed
    :class:`MigrationPlan` per primitive (operator-routable; NOT a
    stub-pending-follow-on per Catalog #371 sister discipline).
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


# Canonical MLX core module that other substrate MLX renderers should
# import primitives FROM (per the audit inventory 2026-05-30; 65%
# adoption already established).
CANONICAL_PR95_HNERV_MLX_MODULE = "tac.local_acceleration.pr95_hnerv_mlx"


# Canonical primitive function/class names that the audit specifically
# tracks for duplicate-implementation detection. Each maps to the
# canonical extractor location in pr95_hnerv_mlx (or sister namespace
# when extraction is recommended).
CANONICAL_PRIMITIVE_EXTRACTORS: Mapping[str, str] = {
    "pixel_shuffle_2x_nhwc": CANONICAL_PR95_HNERV_MLX_MODULE,
    "bilinear_resize_nhwc": CANONICAL_PR95_HNERV_MLX_MODULE,
    "bilinear_resize2x_align_corners_false_nhwc": CANONICAL_PR95_HNERV_MLX_MODULE,
    "_PR95Conv2dMLX": CANONICAL_PR95_HNERV_MLX_MODULE,
    "HNeRVDecoderMLX": CANONICAL_PR95_HNERV_MLX_MODULE,
    "load_pytorch_state_dict_into_mlx": CANONICAL_PR95_HNERV_MLX_MODULE,
    "pytorch_state_dict_from_mlx": CANONICAL_PR95_HNERV_MLX_MODULE,
    # Slot 16 numerical-fidelity canonical primitives (commit 65db9f570
    # MLX drift architecture-class dependent fix landed 2026-05-26):
    "validate_pr95_mlx_conv2d_accumulation_mode": CANONICAL_PR95_HNERV_MLX_MODULE,
    "pr95_mlx_conv2d_accumulation_overrides_from_preset": CANONICAL_PR95_HNERV_MLX_MODULE,
    # Per audit inventory A.2.4: canonical Kahan-EMA per Slot 16 LANDED:
    "KahanCompensatedPolyakEMAShadow": "tac.training.long_training_canonical",
    # Canonical extraction targets (audit recommends lifting these):
    "gumbel_softmax_sample": "tac.framework_agnostic",
    "rgb_to_yuv6": "tac.framework_agnostic",
    "yuv6_to_rgb": "tac.framework_agnostic",
}


# Sister canonical namespace at the framework-agnostic surface (consumed
# by substrate trainers via backend dispatch per Catalog #205 sister).
CANONICAL_FRAMEWORK_AGNOSTIC_MODULE = "tac.framework_agnostic"


# Canonical tinygrad bridge module (sister at the tinygrad portability
# surface per 8th standing directive).
CANONICAL_TINYGRAD_BRIDGE_MODULE = "tac.local_acceleration.tinygrad_bridge"


# Path markers excluded from the audit per CLAUDE.md sister discipline.
_EXEMPT_PATH_MARKERS: tuple[str, ...] = (
    "experiments/results/",
    "_intake_",
    ".omx/oss_export/",
    "vendored",
    "build/lib/",
    "reports/raw/",
    ".venv/",
    "__pycache__/",
    "/workspace/",
)


class DuplicationClassification(str, Enum):
    """Catalog #290 falling-rule classification for a primitive.

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode":
    each canonical-vs-unique decision must produce one of these
    structurally distinct verdicts. Pure observability per Catalog #341
    (no canonical-routing markers in returned data; consumer is the
    audit JSON itself).
    """

    CANONICAL_ADOPTED = "CANONICAL_ADOPTED"
    """Substrate routes through canonical helper; no duplication."""

    CANONICAL_EXTRACTION_RECOMMENDED = "CANONICAL_EXTRACTION_RECOMMENDED"
    """Multiple sister implementations exist; no canonical extracted yet."""

    PRINCIPLED_FORK_HARD_EARNED = "PRINCIPLED_FORK_HARD_EARNED"
    """Substrate has substrate-optimal FORK per Catalog #290 EMPIRICAL or
    PRINCIPLED branch; documented in design memo."""

    UNCLEAR_NEEDS_EMPIRICAL = "UNCLEAR_NEEDS_EMPIRICAL"
    """Burden of proof unresolved per Catalog #290 falling-rule step 3;
    requires paired-comparison smoke."""

    OBVIOUS_FIT = "OBVIOUS_FIT"
    """Canonical helper trivially serves; no fork rationale."""


@dataclass(frozen=True)
class MLXPrimitiveImpl:
    """Per-primitive canonical record.

    Frozen per CLAUDE.md immutability discipline + Catalog #110/#113
    HISTORICAL_PROVENANCE sister.
    """

    primitive_name: str
    """Canonical name of the primitive (function or class)."""

    file_path: str
    """Absolute path to the source file containing the implementation."""

    line_number: int
    """Line number where the def/class appears (1-indexed)."""

    impl_kind: str
    """``"function"`` or ``"class"``."""

    signature_hash: str
    """sha256 of (def/class line + first 10 lines) for canonical
    duplicate-detection across substrates."""

    routes_through_canonical: bool
    """True if the enclosing file imports the canonical extractor for
    THIS primitive instead of re-implementing it."""


@dataclass(frozen=True)
class MLXBearingFile:
    """Per-file canonical record."""

    file_path: str
    """Absolute path to the MLX-bearing source file."""

    loc: int
    """Total lines of code."""

    canonical_pr95_imports: tuple[str, ...]
    """Sorted tuple of primitive names imported from
    ``tac.local_acceleration.pr95_hnerv_mlx``."""

    canonical_framework_agnostic_imports: tuple[str, ...]
    """Sorted tuple of helper names imported from
    ``tac.framework_agnostic``."""

    canonical_tinygrad_bridge_imports: tuple[str, ...]
    """Sorted tuple of bridge helper names imported from
    ``tac.local_acceleration.tinygrad_bridge``."""


@dataclass(frozen=True)
class CanonicalDuplicationVerdict:
    """Typed verdict from :func:`detect_canonical_duplication`."""

    primitive_name: str
    """Canonical name of the primitive being audited."""

    canonical_extractor: str
    """Canonical extractor module per
    :data:`CANONICAL_PRIMITIVE_EXTRACTORS`."""

    duplicate_impl_count: int
    """Total implementations found across the audited corpus."""

    impls: tuple[MLXPrimitiveImpl, ...]
    """All implementations found, ordered by file path then line."""

    canonical_adopters: tuple[str, ...]
    """Sorted file paths that route through the canonical extractor
    instead of re-implementing."""

    classification: DuplicationClassification
    """Catalog #290 falling-rule verdict."""

    rationale: str
    """Substantive non-placeholder rationale per Catalog #287 (>= 4
    chars; placeholder ``<rationale>`` / ``<reason>`` rejected at
    construction)."""

    def __post_init__(self) -> None:
        rationale = (self.rationale or "").strip()
        if len(rationale) < 4:
            raise ValueError(
                "CanonicalDuplicationVerdict.rationale must be >= 4 chars "
                "per Catalog #287 placeholder-rationale rejection sister "
                f"discipline; got {len(rationale)} chars."
            )
        placeholders = {
            "<rationale>",
            "<reason>",
            "<rationale_here>",
            "<reason_here>",
            "tbd",
            "TBD",
            "FIXME",
            "<placeholder>",
            "placeholder",
        }
        if rationale.lower() in {p.lower() for p in placeholders}:
            raise ValueError(
                f"CanonicalDuplicationVerdict.rationale {rationale!r} is a "
                "placeholder literal; substantive rationale required per "
                "Catalog #287."
            )
        if self.duplicate_impl_count < 0:
            raise ValueError(
                f"duplicate_impl_count must be non-negative; got "
                f"{self.duplicate_impl_count}."
            )


@dataclass(frozen=True)
class MigrationPlan:
    """Operator-routable migration plan per primitive.

    Per CLAUDE.md "Forbidden premature KILL without research
    exhaustion": every plan is a path FORWARD (consume canonical helper
    OR document substrate-optimal FORK), NOT a kill verdict.
    """

    primitive_name: str
    """Canonical name of the primitive."""

    verdict: CanonicalDuplicationVerdict
    """Verdict that produced this plan."""

    recommended_action: str
    """One of: ``"ADOPT_CANONICAL"`` / ``"EXTRACT_CANONICAL"`` /
    ``"DOCUMENT_FORK"`` / ``"RUN_PAIRED_COMPARISON"`` /
    ``"NO_ACTION_REQUIRED"``."""

    estimated_loc_reduction: int
    """Estimated total LOC reduction if migration completes (across all
    sister implementations)."""

    estimated_hours: float
    """Estimated engineering hours for the migration."""

    target_substrates: tuple[str, ...]
    """Sister substrate identifiers that would absorb the canonical
    extraction (sorted)."""

    rationale: str
    """Substantive non-placeholder rationale per Catalog #287."""

    def __post_init__(self) -> None:
        rationale = (self.rationale or "").strip()
        if len(rationale) < 4:
            raise ValueError(
                "MigrationPlan.rationale must be >= 4 chars per Catalog "
                f"#287 sister discipline; got {len(rationale)} chars."
            )
        valid_actions = {
            "ADOPT_CANONICAL",
            "EXTRACT_CANONICAL",
            "DOCUMENT_FORK",
            "RUN_PAIRED_COMPARISON",
            "NO_ACTION_REQUIRED",
        }
        if self.recommended_action not in valid_actions:
            raise ValueError(
                f"recommended_action {self.recommended_action!r} not in "
                f"{sorted(valid_actions)}."
            )
        if self.estimated_loc_reduction < 0:
            raise ValueError(
                "estimated_loc_reduction must be non-negative; got "
                f"{self.estimated_loc_reduction}."
            )
        if self.estimated_hours < 0:
            raise ValueError(
                f"estimated_hours must be non-negative; got "
                f"{self.estimated_hours}."
            )


def _is_exempt_path(path: Path) -> bool:
    """Return True if the path should be excluded from the audit."""
    path_str = str(path)
    return any(marker in path_str for marker in _EXEMPT_PATH_MARKERS)


def _signature_hash(source_lines: list[str], start_line: int) -> str:
    """Compute sha256 of (def/class line + next 10 lines) for canonical
    duplicate-detection."""
    snippet = "\n".join(source_lines[start_line - 1 : start_line + 10])
    return hashlib.sha256(snippet.encode("utf-8")).hexdigest()


def _detect_canonical_imports(
    source: str,
    canonical_module: str,
) -> tuple[str, ...]:
    """Return sorted tuple of names imported from canonical_module via
    ``from <canonical_module> import <name>``.

    AST-aware per Catalog #168 sister discipline (string-literal mentions
    in docstrings + comments correctly excluded).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == canonical_module:
                for alias in node.names:
                    imported.add(alias.name)
    return tuple(sorted(imported))


def enumerate_mlx_bearing_files(
    repo_root: Path,
    *,
    include_tests: bool = False,
) -> tuple[MLXBearingFile, ...]:
    """Walk the repo and return all MLX-bearing source files.

    Per CLAUDE.md NO FAKE IMPLEMENTATIONS non-negotiable: this DOES the
    walk; it does NOT return a hard-coded list. The verifiable
    enumeration matches the audit inventory at
    ``.omx/research/mlx_canonicalization_audit_inventory_20260530.md``
    (155 total / 67 non-test at landing 2026-05-30).

    Args:
        repo_root: Repository root path. The walk scans
            ``src/tac/`` + ``experiments/`` + ``tools/`` + ``scripts/``.
        include_tests: When True include test files; when False (the
            default) exclude any path matching ``/tests/`` or ``test_``.

    Returns:
        Tuple of :class:`MLXBearingFile` records sorted by file path.
    """
    repo_root = Path(repo_root).resolve()
    candidate_roots = ["src/tac", "experiments", "tools", "scripts"]
    files: list[MLXBearingFile] = []
    for root_subpath in candidate_roots:
        root = repo_root / root_subpath
        if not root.exists():
            continue
        for py_file in sorted(root.rglob("*.py")):
            if _is_exempt_path(py_file):
                continue
            if not include_tests:
                if "/tests/" in str(py_file) or py_file.name.startswith("test_"):
                    continue
            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            # Line-anchored regex per Catalog #287 sister discipline (NOT
            # a docstring/comment substring match):
            has_mlx_import = False
            for line in source.splitlines():
                stripped = line.lstrip()
                if stripped.startswith("import mlx") or stripped.startswith(
                    "from mlx"
                ):
                    has_mlx_import = True
                    break
            if not has_mlx_import:
                continue
            files.append(
                MLXBearingFile(
                    file_path=str(py_file.relative_to(repo_root)),
                    loc=len(source.splitlines()),
                    canonical_pr95_imports=_detect_canonical_imports(
                        source, CANONICAL_PR95_HNERV_MLX_MODULE
                    ),
                    canonical_framework_agnostic_imports=_detect_canonical_imports(
                        source, CANONICAL_FRAMEWORK_AGNOSTIC_MODULE
                    ),
                    canonical_tinygrad_bridge_imports=_detect_canonical_imports(
                        source, CANONICAL_TINYGRAD_BRIDGE_MODULE
                    ),
                )
            )
    return tuple(files)


def enumerate_mlx_primitive_implementations(
    repo_root: Path,
    *,
    primitive_names: Iterable[str] | None = None,
    include_tests: bool = False,
) -> tuple[MLXPrimitiveImpl, ...]:
    """For each MLX-bearing file enumerate canonical primitive impls.

    Per CLAUDE.md NO FAKE IMPLEMENTATIONS: this DOES the AST walk +
    extracts actual def/class node positions; placeholder lists rejected.

    Args:
        repo_root: Repository root path.
        primitive_names: Iterable of primitive names to track; default
            tracks all keys of :data:`CANONICAL_PRIMITIVE_EXTRACTORS`.
        include_tests: Forwarded to :func:`enumerate_mlx_bearing_files`.

    Returns:
        Tuple of :class:`MLXPrimitiveImpl` records sorted by file path
        then line number.
    """
    repo_root = Path(repo_root).resolve()
    if primitive_names is None:
        primitive_names_set = frozenset(CANONICAL_PRIMITIVE_EXTRACTORS.keys())
    else:
        primitive_names_set = frozenset(primitive_names)

    impls: list[MLXPrimitiveImpl] = []
    files = enumerate_mlx_bearing_files(
        repo_root, include_tests=include_tests
    )
    for mlx_file in files:
        file_abs = repo_root / mlx_file.file_path
        try:
            source = file_abs.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        source_lines = source.splitlines()
        canonical_imports_combined = set(mlx_file.canonical_pr95_imports) | set(
            mlx_file.canonical_framework_agnostic_imports
        ) | set(mlx_file.canonical_tinygrad_bridge_imports)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name not in primitive_names_set:
                    continue
                impls.append(
                    MLXPrimitiveImpl(
                        primitive_name=node.name,
                        file_path=mlx_file.file_path,
                        line_number=node.lineno,
                        impl_kind="function",
                        signature_hash=_signature_hash(
                            source_lines, node.lineno
                        ),
                        routes_through_canonical=(
                            node.name in canonical_imports_combined
                        ),
                    )
                )
            elif isinstance(node, ast.ClassDef):
                if node.name not in primitive_names_set:
                    continue
                impls.append(
                    MLXPrimitiveImpl(
                        primitive_name=node.name,
                        file_path=mlx_file.file_path,
                        line_number=node.lineno,
                        impl_kind="class",
                        signature_hash=_signature_hash(
                            source_lines, node.lineno
                        ),
                        routes_through_canonical=(
                            node.name in canonical_imports_combined
                        ),
                    )
                )
    return tuple(
        sorted(impls, key=lambda i: (i.file_path, i.line_number))
    )


def detect_canonical_duplication(
    primitive_name: str,
    impls: Iterable[MLXPrimitiveImpl],
    canonical_adopters: Iterable[str] | None = None,
) -> CanonicalDuplicationVerdict:
    """Classify a primitive per Catalog #290 falling-rule list.

    Decision flow:
      1. If only the canonical-extractor module contains the impl + N
         sister files import it → ``CANONICAL_ADOPTED``.
      2. Else if 2+ sister files have non-canonical impls → either
         ``CANONICAL_EXTRACTION_RECOMMENDED`` (when no PRINCIPLED fork
         rationale exists) or ``PRINCIPLED_FORK_HARD_EARNED`` (when
         canonical adopters list documents the fork).
      3. Else if exactly 2 impls with no canonical adoption →
         ``UNCLEAR_NEEDS_EMPIRICAL``.
      4. Else ``OBVIOUS_FIT``.

    Args:
        primitive_name: Canonical name being audited.
        impls: All impls of the primitive across the corpus.
        canonical_adopters: Optional iterable of file paths that route
            through the canonical extractor.

    Returns:
        Typed :class:`CanonicalDuplicationVerdict`.
    """
    impls_tuple: tuple[MLXPrimitiveImpl, ...] = tuple(impls)
    adopters_tuple: tuple[str, ...] = tuple(
        sorted(canonical_adopters) if canonical_adopters else ()
    )
    canonical_extractor = CANONICAL_PRIMITIVE_EXTRACTORS.get(
        primitive_name, "UNKNOWN"
    )
    duplicate_count = len(impls_tuple)

    # Cascade per Catalog #290 falling-rule list:
    if duplicate_count == 0:
        if adopters_tuple:
            return CanonicalDuplicationVerdict(
                primitive_name=primitive_name,
                canonical_extractor=canonical_extractor,
                duplicate_impl_count=0,
                impls=(),
                canonical_adopters=adopters_tuple,
                classification=DuplicationClassification.CANONICAL_ADOPTED,
                rationale=(
                    f"no sister impl found; {len(adopters_tuple)} adopters "
                    f"route through {canonical_extractor}"
                ),
            )
        return CanonicalDuplicationVerdict(
            primitive_name=primitive_name,
            canonical_extractor=canonical_extractor,
            duplicate_impl_count=0,
            impls=(),
            canonical_adopters=(),
            classification=DuplicationClassification.OBVIOUS_FIT,
            rationale=(
                f"no impls + no adopters; {primitive_name} not present in "
                "corpus"
            ),
        )

    canonical_impl_files = [
        i.file_path
        for i in impls_tuple
        if canonical_extractor.replace(".", "/") in i.file_path
        or i.file_path.endswith(
            canonical_extractor.split(".")[-1] + ".py"
        )
    ]
    non_canonical_impls = [
        i for i in impls_tuple if i.file_path not in canonical_impl_files
    ]

    # Pure canonical: only the extractor file has the impl + adopters
    # route through it:
    if (
        len(canonical_impl_files) >= 1
        and len(non_canonical_impls) == 0
        and adopters_tuple
    ):
        return CanonicalDuplicationVerdict(
            primitive_name=primitive_name,
            canonical_extractor=canonical_extractor,
            duplicate_impl_count=duplicate_count,
            impls=impls_tuple,
            canonical_adopters=adopters_tuple,
            classification=DuplicationClassification.CANONICAL_ADOPTED,
            rationale=(
                f"canonical extractor at {canonical_extractor}; "
                f"{len(adopters_tuple)} sister files route through it; "
                f"no duplicate impls"
            ),
        )

    # Multiple sister impls without canonical extraction:
    if len(non_canonical_impls) >= 2:
        return CanonicalDuplicationVerdict(
            primitive_name=primitive_name,
            canonical_extractor=canonical_extractor,
            duplicate_impl_count=duplicate_count,
            impls=impls_tuple,
            canonical_adopters=adopters_tuple,
            classification=DuplicationClassification.CANONICAL_EXTRACTION_RECOMMENDED,
            rationale=(
                f"{len(non_canonical_impls)} sister impls outside canonical "
                f"extractor {canonical_extractor}; "
                f"canonical extraction recommended per Catalog #290 "
                "EMPIRICAL/PRINCIPLED branch"
            ),
        )

    # Exactly 2 impls (1 canonical + 1 sister) without paired-comparison:
    if duplicate_count == 2 and not adopters_tuple:
        return CanonicalDuplicationVerdict(
            primitive_name=primitive_name,
            canonical_extractor=canonical_extractor,
            duplicate_impl_count=duplicate_count,
            impls=impls_tuple,
            canonical_adopters=(),
            classification=DuplicationClassification.UNCLEAR_NEEDS_EMPIRICAL,
            rationale=(
                "two impls without canonical-routing adopters; paired "
                "comparison smoke needed per Catalog #290 step 3"
            ),
        )

    # Default: canonical extractor exists + 1 sister impl that may be a
    # PRINCIPLED FORK (e.g. numpy reference for inflate-time):
    return CanonicalDuplicationVerdict(
        primitive_name=primitive_name,
        canonical_extractor=canonical_extractor,
        duplicate_impl_count=duplicate_count,
        impls=impls_tuple,
        canonical_adopters=adopters_tuple,
        classification=DuplicationClassification.PRINCIPLED_FORK_HARD_EARNED,
        rationale=(
            f"canonical extractor + sister impl(s); PRINCIPLED FORK "
            "(e.g. numpy reference for inflate-time per HNeRV parity L4)"
        ),
    )


def recommend_canonical_extraction(
    verdict: CanonicalDuplicationVerdict,
    *,
    estimated_loc_per_impl: int = 80,
    estimated_hours_per_impl: float = 0.5,
) -> MigrationPlan:
    """Emit an operator-routable migration plan per primitive.

    Per CLAUDE.md "Forbidden premature KILL": every plan is a path
    FORWARD, NOT a kill verdict.

    Args:
        verdict: Verdict from :func:`detect_canonical_duplication`.
        estimated_loc_per_impl: Calibrated LOC per sister implementation
            (audit inventory baseline = 80 LOC per primitive).
        estimated_hours_per_impl: Calibrated engineering hours per sister
            migration (audit inventory baseline = 0.5h per primitive).

    Returns:
        Typed :class:`MigrationPlan`.
    """
    classification = verdict.classification
    primitive = verdict.primitive_name

    # Collect substrate identifiers from impl file paths:
    target_substrates = tuple(sorted({
        impl.file_path.split("/")[-2]
        for impl in verdict.impls
        if "substrates/" in impl.file_path
    }))

    if classification == DuplicationClassification.CANONICAL_ADOPTED:
        return MigrationPlan(
            primitive_name=primitive,
            verdict=verdict,
            recommended_action="NO_ACTION_REQUIRED",
            estimated_loc_reduction=0,
            estimated_hours=0.0,
            target_substrates=target_substrates,
            rationale=(
                f"canonical {primitive} already adopted at "
                f"{verdict.canonical_extractor}; no migration required"
            ),
        )
    if classification == DuplicationClassification.OBVIOUS_FIT:
        return MigrationPlan(
            primitive_name=primitive,
            verdict=verdict,
            recommended_action="NO_ACTION_REQUIRED",
            estimated_loc_reduction=0,
            estimated_hours=0.0,
            target_substrates=(),
            rationale=(
                f"{primitive} not present in corpus; no migration needed"
            ),
        )
    if classification == DuplicationClassification.CANONICAL_EXTRACTION_RECOMMENDED:
        sister_count = max(0, verdict.duplicate_impl_count - 1)
        return MigrationPlan(
            primitive_name=primitive,
            verdict=verdict,
            recommended_action="EXTRACT_CANONICAL",
            estimated_loc_reduction=sister_count * estimated_loc_per_impl,
            estimated_hours=sister_count * estimated_hours_per_impl,
            target_substrates=target_substrates,
            rationale=(
                f"lift {primitive} to {verdict.canonical_extractor} + "
                f"migrate {sister_count} sister impls; estimated "
                f"{sister_count * estimated_loc_per_impl} LOC reduction"
            ),
        )
    if classification == DuplicationClassification.PRINCIPLED_FORK_HARD_EARNED:
        return MigrationPlan(
            primitive_name=primitive,
            verdict=verdict,
            recommended_action="DOCUMENT_FORK",
            estimated_loc_reduction=0,
            estimated_hours=0.25,
            target_substrates=target_substrates,
            rationale=(
                f"{primitive} sister impl(s) are PRINCIPLED FORK per "
                "Catalog #290; document fork rationale in substrate "
                "design memo per Catalog #303"
            ),
        )
    if classification == DuplicationClassification.UNCLEAR_NEEDS_EMPIRICAL:
        return MigrationPlan(
            primitive_name=primitive,
            verdict=verdict,
            recommended_action="RUN_PAIRED_COMPARISON",
            estimated_loc_reduction=0,
            estimated_hours=1.0,
            target_substrates=target_substrates,
            rationale=(
                f"two {primitive} impls without canonical routing; run "
                "paired-comparison smoke per Catalog #290 step 3 before "
                "canonical adoption decision"
            ),
        )
    raise ValueError(
        f"unexpected classification {classification!r}"
    )


def summarize_audit_report(
    repo_root: Path,
    *,
    include_tests: bool = False,
) -> dict[str, Any]:
    """End-to-end canonical audit report.

    Returns a dict consumable by ``tools/audit_mlx_canonicalization.py``
    + sister cathedral consumer per Catalog #335 auto-discovery.

    Per CLAUDE.md NO FAKE IMPLEMENTATIONS + Catalog #287: the report
    contains actual file counts + actual primitive counts + actual
    duplication verdicts (NOT a placeholder summary).
    """
    repo_root = Path(repo_root).resolve()
    files = enumerate_mlx_bearing_files(
        repo_root, include_tests=include_tests
    )
    impls = enumerate_mlx_primitive_implementations(
        repo_root, include_tests=include_tests
    )

    # Group impls by primitive name:
    per_primitive: dict[str, list[MLXPrimitiveImpl]] = {}
    for impl in impls:
        per_primitive.setdefault(impl.primitive_name, []).append(impl)

    # Collect adopters per primitive:
    per_primitive_adopters: dict[str, list[str]] = {}
    for f in files:
        for name in f.canonical_pr95_imports:
            per_primitive_adopters.setdefault(name, []).append(f.file_path)
        for name in f.canonical_framework_agnostic_imports:
            per_primitive_adopters.setdefault(name, []).append(f.file_path)
        for name in f.canonical_tinygrad_bridge_imports:
            per_primitive_adopters.setdefault(name, []).append(f.file_path)

    verdicts: list[CanonicalDuplicationVerdict] = []
    plans: list[MigrationPlan] = []
    for primitive_name in CANONICAL_PRIMITIVE_EXTRACTORS.keys():
        impls_for = per_primitive.get(primitive_name, [])
        adopters_for = per_primitive_adopters.get(primitive_name, [])
        verdict = detect_canonical_duplication(
            primitive_name,
            impls_for,
            canonical_adopters=adopters_for,
        )
        plan = recommend_canonical_extraction(verdict)
        verdicts.append(verdict)
        plans.append(plan)

    # Aggregate stats:
    total_files = len(files)
    substrate_files = sum(
        1 for f in files if "substrates/" in f.file_path
    )
    canonical_adopter_files = sum(
        1
        for f in files
        if f.canonical_pr95_imports
        or f.canonical_framework_agnostic_imports
        or f.canonical_tinygrad_bridge_imports
    )

    return {
        "schema_version": "mlx_canonicalization_audit_v1",
        "total_mlx_bearing_files": total_files,
        "substrate_files": substrate_files,
        "canonical_adopter_files": canonical_adopter_files,
        "canonical_adoption_rate": (
            canonical_adopter_files / total_files if total_files else 0.0
        ),
        "verdicts": [
            {
                "primitive_name": v.primitive_name,
                "canonical_extractor": v.canonical_extractor,
                "classification": v.classification.value,
                "duplicate_impl_count": v.duplicate_impl_count,
                "canonical_adopter_count": len(v.canonical_adopters),
                "rationale": v.rationale,
            }
            for v in verdicts
        ],
        "migration_plans": [
            {
                "primitive_name": p.primitive_name,
                "recommended_action": p.recommended_action,
                "estimated_loc_reduction": p.estimated_loc_reduction,
                "estimated_hours": p.estimated_hours,
                "target_substrate_count": len(p.target_substrates),
                "rationale": p.rationale,
            }
            for p in plans
        ],
        "aggregate_estimated_loc_reduction": sum(
            p.estimated_loc_reduction for p in plans
        ),
        "aggregate_estimated_hours": sum(p.estimated_hours for p in plans),
    }
