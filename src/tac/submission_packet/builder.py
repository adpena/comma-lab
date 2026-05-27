# SPDX-License-Identifier: MIT
"""Layer 2 — canonical inflate-runtime bundler.

Wrap per-substrate ``submission_dir/`` emission into one canonical entry point
per Phase 1 audit specification memo at
``.omx/research/canonical_submission_pipeline_specification_memo_20260526.md``
Layer 2 (largest single layer per the spec; ~600-900 LOC).

The bug class this layer extincts: ad-hoc per-substrate ``submission_dir/``
builders that drift on (i) HNeRV parity L4 ``inflate.py`` ≤200 LOC + ≤2
external deps + numpy-portable + CUDA-or-CPU agnostic + reviewable in 30s,
(ii) Catalog #205 canonical ``select_inflate_device`` routing (no inline
device-fork without explicit waiver), (iii) Catalog #295 PYTHONPATH
self-containment (no bare ``from tac.*`` without a vendored sister package),
(iv) Catalog #146 contest-compliant ``inflate.sh`` 3-arg signature
(``archive_dir`` / ``output_dir`` / ``file_list``), (v) Catalog #361 Modal
artifact filter compatibility (``output/submission/`` subtree mtime-fresh),
(vi) Catalog #208 docs no-local-absolute-paths sanitization, (vii) report.txt
+ README.md attribution chain shape per PR 95 medal-class precedent.

Per the 8th MLX-first numpy-portable individually-fractal standing directive:
the bundler EMITS a numpy-only inflate runtime. Encoder training is MLX-first
on Apple Silicon; bundled ``inflate.py`` carries ONLY numpy + pyav (no MLX,
no torch dependency for the canonical-numpy-portable path). Submissions that
require torch at inflate time (HNeRV-class current PR101 baseline) declare
``inflate_deps_budget=2`` and carry ``torch`` + ``numpy`` as the two deps;
that is honored as canonical-but-not-numpy-portable per the contract.

Per the 11th ORDER-MATTERS standing directive: Layer 2 (this module) is the
THIRD Phase 1 spec consumer; depends on Layer 0
:class:`tac.submission_packet.compression_pipeline.CompressionPipelineResult`
and Layer 1
:class:`tac.submission_packet.archive_grammar.ArchiveGrammarManifest`
dataclass shapes; downstream Phase 5-10 layers depend on this module's
:class:`SubmissionBundleResult` shape.

Per the 12th canonicalization × standardization × ease-of-contest-compliance
trinity: ONE canonical helper, ONE return shape, ONE bundle-emission
protocol.

Per Catalog #341 + CLAUDE.md "Apples-to-apples evidence discipline": this
layer is OBSERVABILITY-ONLY by construction. Every emitted
:class:`SubmissionBundleResult` carries ``score_claim=False`` +
``promotable=False`` + ``axis_tag=[predicted]``. Promotion of a bundled
submission to a contest score signal REQUIRES Phase 6 paired-CUDA + Linux
x86_64 CPU empirical anchor per CLAUDE.md "Submission auth eval — BOTH CPU
AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable.
"""
from __future__ import annotations

import datetime
import enum
import hashlib
import os
import re
import socket
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tac.submission_packet.archive_grammar import ArchiveGrammarManifest
from tac.submission_packet.compression_pipeline import (
    CompressionPipelineResult,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Module-level constants — canonical schemas + canonical budgets
# ---------------------------------------------------------------------------

SUBMISSION_BUNDLE_SCHEMA_VERSION = "submission_bundle_v1_20260526"
"""Pinned schema for :class:`SubmissionBundleResult` persistence rows."""

PHASE_4_LAYER_VERSION = "phase_4_submission_bundle_canonical_landed_20260526"
"""Operator-readable Phase 4 landing marker per Phase 1 audit spec memo."""

CANONICAL_EQUATION_ID = (
    "submission_bundle_canonical_helper_consolidation_savings_v1"
)
"""Canonical equation registered per Phase 1 audit spec memo §13.

FORMALIZATION_PENDING until Phase 10 first-PR-through-canonical-pipeline
regression lands the first paired-CUDA empirical anchor of per-substrate
bundle-divergence collapse (predicted: 14 per-substrate ad-hoc
submission_dir builders consolidated to ONE canonical helper).
"""

# Per Catalog #341 routing markers (Tier A observability-only).
PREDICTED_AXIS_TAG = "[predicted]"

# Per Catalog #287 placeholder rejection.
_PLACEHOLDER_RATIONALES: frozenset[str] = frozenset(
    {"<rationale>", "<reason>", "<rationale_here>", "<reason_here>", ""}
)

# Per HNeRV parity L4 — inflate.py LOC + dep budgets (NON-NEGOTIABLE
# default; explicit waiver required to exceed).
DEFAULT_INFLATE_PY_LOC_BUDGET = 200
DEFAULT_INFLATE_DEPS_BUDGET = 2

# Canonical numpy-portable inflate dep set per 8th MLX-first standing directive.
NUMPY_PORTABLE_INFLATE_DEPS: frozenset[str] = frozenset({"numpy"})

# Canonical HNeRV-class inflate dep set (≤2 deps per HNeRV parity L4).
HNERV_CLASS_INFLATE_DEPS: frozenset[str] = frozenset({"torch", "numpy"})

# Canonical 3-arg inflate.sh signature per Catalog #146 contest-compliant
# runtime template.
CANONICAL_INFLATE_SH_POSITIONAL_TOKENS: tuple[str, ...] = (
    "$1",
    "$2",
    "$3",
)
CANONICAL_INFLATE_SH_NAMED_TOKENS: tuple[str, ...] = (
    "DATA_DIR",
    "OUTPUT_DIR",
    "FILE_LIST",
)
CANONICAL_INFLATE_SH_REQUIRED_HEADER = "set -euo pipefail"

# Per Catalog #208 — local-absolute-path patterns forbidden in submission_dir
# README + report.txt + other docs surfaces.
_LOCAL_ABSOLUTE_PATH_PATTERNS: tuple[str, ...] = (
    r"/Users/\w+/",
    r"/home/\w+/",
    r"/private/var/",
    r"C:\\Users\\\w+\\",
)

# Per CLAUDE.md "Public Disclosure Hygiene" + per the standing directive at
# feedback_forbidden_claude_attribution_in_public_pr_surfaces.md.
_FORBIDDEN_PUBLIC_PR_TOKENS: tuple[str, ...] = (
    "Claude",
    "Anthropic",
    "Co-Authored",
    "claude.com",
    "anthropic.com",
)


# Canonical bundle component kinds.
class BundleComponentKind(enum.StrEnum):
    """Canonical submission_dir component taxonomy.

    Per HNeRV parity L4: every submission_dir MUST carry at minimum the
    inflate runtime (inflate.sh + inflate.py), the archive bytes
    (archive.zip), the human-readable README.md, and the report.txt
    placeholder. Additional sidecars (manifests, evidence JSONs, vendored
    sister packages per Catalog #295) are kind-specific.
    """

    INFLATE_SH = "inflate_sh"
    INFLATE_PY = "inflate_py"
    ARCHIVE_ZIP = "archive_zip"
    README_MD = "readme_md"
    REPORT_TXT = "report_txt"
    ARCHIVE_MANIFEST = "archive_manifest"
    VENDORED_PACKAGE = "vendored_package"
    SIDECAR_EVIDENCE = "sidecar_evidence"
    OTHER = "other"


# Canonical inflate-device routing per Catalog #205.
class SelectInflateDeviceRouting(enum.StrEnum):
    """How the bundled inflate.py routes the canonical select_inflate_device."""

    CANONICAL_HELPER = "canonical_helper"
    """Uses ``tac.substrates._shared.inflate_runtime.select_inflate_device``
    (requires vendored package per Catalog #295)."""

    INLINE_WITH_WAIVER = "inline_with_waiver"
    """Inlined byte-identical mirror per Catalog #205 + INLINE_DEVICE_FORK_OK
    waiver pattern (canonical default for sole-runtime submission_dirs)."""


# Canonical PYTHONPATH self-containment status per Catalog #295.
class PythonpathSelfContainmentStatus(enum.StrEnum):
    """How the bundled inflate.py satisfies Catalog #295 self-containment."""

    CLEAN = "clean"
    """No bare ``from tac.*`` imports; inflate is fully self-contained."""

    VENDORED_WITH_EXPLICIT_WAIVER = "vendored_with_explicit_waiver"
    """``from tac.*`` import is satisfied via vendored sister package
    alongside (e.g. ``submission_dir/src/tac/...``) per Catalog #295."""

    SCAFFOLD_PENDING = "scaffold_pending"
    """Bundle emitted in scaffold mode; PYTHONPATH self-containment
    deferred until Phase 5/6 dispatch-ready landing."""


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class SubmissionBundleError(RuntimeError):
    """Submission bundle orchestration error.

    Sister of :class:`tac.submission_packet.archive_grammar.ArchiveGrammarError`
    + :class:`tac.submission_packet.compression_pipeline.CompressionPipelineError`.
    Raised by :func:`build_submission_bundle` when HNeRV parity L4 invariants
    cannot be satisfied AND no waiver is supplied.
    """


# ---------------------------------------------------------------------------
# Frozen dataclasses — canonical contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DependencyClosureManifest:
    """Per-bundle dependency closure manifest per HNeRV parity L4.

    Declares the FULL set of external dependencies inflate.py requires at
    runtime (canonical: ≤2 deps per HNeRV parity L4). Sister of canonical
    Provenance per Catalog #323.
    """

    declared_dependencies: tuple[str, ...]
    """Canonical sorted tuple of dependency names (e.g. ``("numpy",)`` or
    ``("numpy", "torch")``)."""

    dependency_budget: int
    """Operator-set budget (canonical default :data:`DEFAULT_INFLATE_DEPS_BUDGET`)."""

    within_budget: bool
    """True iff ``len(declared_dependencies) <= dependency_budget``."""

    numpy_portable: bool
    """True iff ``declared_dependencies == ("numpy",)`` (canonical 8th
    standing directive numpy-portable inflate)."""

    waiver_rationale: str | None = None
    """Substantive rationale (≥4 chars, non-placeholder per Catalog #287)
    when ``within_budget=False``."""

    def __post_init__(self) -> None:
        if not isinstance(self.declared_dependencies, tuple):
            raise ValueError("declared_dependencies must be a tuple (frozen)")
        for dep in self.declared_dependencies:
            if not isinstance(dep, str) or not dep.strip():
                raise ValueError(
                    f"declared_dependencies entries must be non-empty strings; got {dep!r}"
                )
        sorted_deps = tuple(sorted(self.declared_dependencies))
        if sorted_deps != self.declared_dependencies:
            raise ValueError(
                "declared_dependencies must be sorted canonical tuple; "
                f"got {self.declared_dependencies}; canonical {sorted_deps}"
            )
        if self.dependency_budget < 0:
            raise ValueError("dependency_budget must be non-negative")
        if not isinstance(self.within_budget, bool):
            raise ValueError("within_budget must be bool")
        expected_within = len(self.declared_dependencies) <= self.dependency_budget
        if self.within_budget != expected_within:
            raise ValueError(
                f"within_budget {self.within_budget} inconsistent with "
                f"len(declared_dependencies)={len(self.declared_dependencies)} "
                f"vs dependency_budget={self.dependency_budget}"
            )
        if not isinstance(self.numpy_portable, bool):
            raise ValueError("numpy_portable must be bool")
        expected_numpy_portable = self.declared_dependencies == ("numpy",)
        if self.numpy_portable != expected_numpy_portable:
            raise ValueError(
                f"numpy_portable {self.numpy_portable} inconsistent with "
                f"declared_dependencies={self.declared_dependencies}"
            )
        if not self.within_budget:
            if self.waiver_rationale is None:
                raise ValueError(
                    "within_budget=False requires non-None waiver_rationale per Catalog #287"
                )
            stripped = self.waiver_rationale.strip()
            if stripped in _PLACEHOLDER_RATIONALES or len(stripped) < 4:
                raise ValueError(
                    f"waiver_rationale {self.waiver_rationale!r} must be substantive "
                    "(>=4 chars, non-placeholder) per Catalog #287"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "declared_dependencies": list(self.declared_dependencies),
            "dependency_budget": int(self.dependency_budget),
            "within_budget": bool(self.within_budget),
            "numpy_portable": bool(self.numpy_portable),
            "waiver_rationale": self.waiver_rationale,
        }


@dataclass(frozen=True)
class SubmissionBundleResult:
    """Canonical Phase 4 Layer 2 submission bundle output.

    Sister of :class:`tac.submission_packet.compression_pipeline.CompressionPipelineResult`
    + :class:`tac.submission_packet.archive_grammar.ArchiveGrammarManifest`
    at the submission-packet bundling sub-surface.

    Per HNeRV parity L4: the bundled inflate.py MUST be ≤200 LOC AND carry
    ≤2 external deps AND be numpy-portable (CUDA-or-CPU agnostic, no MPS)
    AND be reviewable in 30 seconds. Default-mode build refuses to emit a
    bundle that violates these invariants; an explicit
    ``inflate_loc_budget_waiver`` rationale (≥4 chars, non-placeholder per
    Catalog #287) is required to exceed the LOC budget.
    """

    schema_version: str
    """Canonical schema version (current: :data:`SUBMISSION_BUNDLE_SCHEMA_VERSION`)."""

    lane_id: str
    """Lane registry id from compression pipeline lineage."""

    substrate_id: str
    """Substrate id from compression pipeline lineage."""

    archive_sha256: str
    """sha256 hex digest of ``submission_dir/archive.zip`` (canonical regression
    surface; MUST match ArchiveGrammarManifest.archive_sha256)."""

    archive_bytes: int
    """Total ``archive.zip`` size in bytes."""

    submission_dir: str
    """Absolute or repo-relative path to the bundled ``submission_dir/``."""

    inflate_sh_path: str
    """Path to the bundled ``inflate.sh`` per Catalog #146 3-arg signature."""

    inflate_py_path: str
    """Path to the bundled ``inflate.py`` per HNeRV parity L4."""

    inflate_py_loc: int
    """Physical LOC of bundled ``inflate.py`` (must be ≤
    :data:`DEFAULT_INFLATE_PY_LOC_BUDGET` OR carry a substantive waiver)."""

    inflate_py_loc_budget: int
    """Operator-set LOC budget (canonical default
    :data:`DEFAULT_INFLATE_PY_LOC_BUDGET`)."""

    inflate_py_loc_waiver_rationale: str | None
    """Substantive rationale when ``inflate_py_loc > inflate_py_loc_budget``."""

    readme_md_path: str
    """Path to the bundled ``README.md`` with attribution chain placeholder."""

    report_txt_path: str
    """Path to the bundled ``report.txt`` placeholder (filled by Phase 6 paired-auth-eval)."""

    archive_manifest_path: str
    """Path to the bundled ``archive_manifest.json`` sidecar (canonical per-member identity)."""

    dependency_closure_manifest: DependencyClosureManifest
    """Per-bundle dependency closure per HNeRV parity L4."""

    select_inflate_device_routing: str
    """One of :class:`SelectInflateDeviceRouting` values."""

    pythonpath_self_containment_status: str
    """One of :class:`PythonpathSelfContainmentStatus` values."""

    vendor_pythonpath_self_containment: bool
    """Caller-set flag: when True, vendor any ``tac.*`` deps alongside
    inflate.py per Catalog #295."""

    runtime_dep_closure: tuple[str, ...]
    """Canonical sorted tuple of all runtime deps (alias of
    ``dependency_closure_manifest.declared_dependencies``; preserved for
    backward-compat with Phase 5 compliance consumer)."""

    measurement_utc: str
    """ISO-8601 UTC timestamp of bundle emission."""

    axis_tag: str
    """Always ``"[predicted]"`` per Catalog #341 + canonical Provenance."""

    score_claim: bool
    """Always ``False`` per CLAUDE.md "Apples-to-apples evidence discipline"."""

    promotable: bool
    """Always ``False`` per Catalog #341 + #192."""

    evidence_grade: str
    """Always ``"[predicted; submission-bundle-canonical]"`` per Catalog #287/#323."""

    canonical_helper_invocation: str
    """``"tac.submission_packet.build_submission_bundle"`` per Catalog #190."""

    canonical_equation_id: str
    """:data:`CANONICAL_EQUATION_ID` (per Catalog #344)."""

    canonical_equation_status: str
    """``"FORMALIZATION_PENDING"`` until Phase 10 first empirical anchor."""

    elapsed_seconds: float
    """Bundle-build elapsed wall-clock."""

    canonical_provenance: Mapping[str, Any] = field(default_factory=dict)
    """Per Catalog #323 canonical Provenance umbrella."""

    written_at_utc: str = ""
    """When persisted to a canonical ledger (caller-fills)."""

    written_pid: int = 0
    """Process PID that emitted the result."""

    written_host: str = ""
    """Host that emitted the result."""

    def __post_init__(self) -> None:
        if self.schema_version != SUBMISSION_BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must equal {SUBMISSION_BUNDLE_SCHEMA_VERSION!r}; "
                f"got {self.schema_version!r}"
            )
        if not self.lane_id:
            raise ValueError("lane_id must be non-empty")
        if not self.substrate_id:
            raise ValueError("substrate_id must be non-empty")
        if len(self.archive_sha256) != 64:
            raise ValueError(
                f"archive_sha256 must be 64-char hex; got len={len(self.archive_sha256)}"
            )
        if self.archive_bytes < 0:
            raise ValueError("archive_bytes must be non-negative")
        for label, value in (
            ("submission_dir", self.submission_dir),
            ("inflate_sh_path", self.inflate_sh_path),
            ("inflate_py_path", self.inflate_py_path),
            ("readme_md_path", self.readme_md_path),
            ("report_txt_path", self.report_txt_path),
            ("archive_manifest_path", self.archive_manifest_path),
        ):
            if not value:
                raise ValueError(f"{label} must be non-empty")
        if self.inflate_py_loc < 0:
            raise ValueError("inflate_py_loc must be non-negative")
        if self.inflate_py_loc_budget < 0:
            raise ValueError("inflate_py_loc_budget must be non-negative")
        # HNeRV parity L4: refuse over-budget without substantive waiver.
        if self.inflate_py_loc > self.inflate_py_loc_budget:
            if self.inflate_py_loc_waiver_rationale is None:
                raise ValueError(
                    f"inflate_py_loc={self.inflate_py_loc} exceeds budget "
                    f"{self.inflate_py_loc_budget} per HNeRV parity L4; "
                    "non-None inflate_py_loc_waiver_rationale required"
                )
            stripped = self.inflate_py_loc_waiver_rationale.strip()
            if stripped in _PLACEHOLDER_RATIONALES or len(stripped) < 4:
                raise ValueError(
                    f"inflate_py_loc_waiver_rationale {self.inflate_py_loc_waiver_rationale!r} "
                    "must be substantive (>=4 chars, non-placeholder) per Catalog #287"
                )
        if not isinstance(self.dependency_closure_manifest, DependencyClosureManifest):
            raise ValueError(
                "dependency_closure_manifest must be a DependencyClosureManifest instance"
            )
        if self.select_inflate_device_routing not in {
            r.value for r in SelectInflateDeviceRouting
        }:
            raise ValueError(
                f"select_inflate_device_routing {self.select_inflate_device_routing!r} "
                f"must be one of {[r.value for r in SelectInflateDeviceRouting]} per Catalog #205"
            )
        if self.pythonpath_self_containment_status not in {
            s.value for s in PythonpathSelfContainmentStatus
        }:
            raise ValueError(
                f"pythonpath_self_containment_status "
                f"{self.pythonpath_self_containment_status!r} must be one of "
                f"{[s.value for s in PythonpathSelfContainmentStatus]} per Catalog #295"
            )
        if not isinstance(self.vendor_pythonpath_self_containment, bool):
            raise ValueError("vendor_pythonpath_self_containment must be bool")
        if not isinstance(self.runtime_dep_closure, tuple):
            raise ValueError("runtime_dep_closure must be a tuple (frozen)")
        if (
            tuple(self.runtime_dep_closure)
            != tuple(self.dependency_closure_manifest.declared_dependencies)
        ):
            raise ValueError(
                "runtime_dep_closure must equal dependency_closure_manifest.declared_dependencies"
            )
        if self.axis_tag != PREDICTED_AXIS_TAG:
            raise ValueError(f"axis_tag must equal {PREDICTED_AXIS_TAG!r}; got {self.axis_tag!r}")
        if self.score_claim is not False:
            raise ValueError("score_claim must be False per Catalog #341")
        if self.promotable is not False:
            raise ValueError("promotable must be False per Catalog #341")
        if not self.evidence_grade.startswith("[predicted;"):
            raise ValueError(
                "evidence_grade must start with '[predicted;' per Catalog #287/#323"
            )
        if self.canonical_equation_id != CANONICAL_EQUATION_ID:
            raise ValueError(
                f"canonical_equation_id must equal {CANONICAL_EQUATION_ID!r}; "
                f"got {self.canonical_equation_id!r}"
            )
        if self.canonical_equation_status not in {"FORMALIZATION_PENDING", "REGISTERED"}:
            raise ValueError(
                "canonical_equation_status must be 'FORMALIZATION_PENDING' or 'REGISTERED' per Catalog #344"
            )
        if not self.measurement_utc:
            raise ValueError("measurement_utc must be non-empty")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        if not isinstance(self.canonical_provenance, Mapping):
            raise ValueError("canonical_provenance must be a Mapping per Catalog #323")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lane_id": self.lane_id,
            "substrate_id": self.substrate_id,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": int(self.archive_bytes),
            "submission_dir": self.submission_dir,
            "inflate_sh_path": self.inflate_sh_path,
            "inflate_py_path": self.inflate_py_path,
            "inflate_py_loc": int(self.inflate_py_loc),
            "inflate_py_loc_budget": int(self.inflate_py_loc_budget),
            "inflate_py_loc_waiver_rationale": self.inflate_py_loc_waiver_rationale,
            "readme_md_path": self.readme_md_path,
            "report_txt_path": self.report_txt_path,
            "archive_manifest_path": self.archive_manifest_path,
            "dependency_closure_manifest": self.dependency_closure_manifest.as_dict(),
            "select_inflate_device_routing": self.select_inflate_device_routing,
            "pythonpath_self_containment_status": self.pythonpath_self_containment_status,
            "vendor_pythonpath_self_containment": bool(
                self.vendor_pythonpath_self_containment
            ),
            "runtime_dep_closure": list(self.runtime_dep_closure),
            "measurement_utc": self.measurement_utc,
            "axis_tag": self.axis_tag,
            "score_claim": bool(self.score_claim),
            "promotable": bool(self.promotable),
            "evidence_grade": self.evidence_grade,
            "canonical_helper_invocation": self.canonical_helper_invocation,
            "canonical_equation_id": self.canonical_equation_id,
            "canonical_equation_status": self.canonical_equation_status,
            "elapsed_seconds": float(self.elapsed_seconds),
            "canonical_provenance": dict(self.canonical_provenance),
            "written_at_utc": self.written_at_utc,
            "written_pid": int(self.written_pid),
            "written_host": self.written_host,
        }


# ---------------------------------------------------------------------------
# Core API helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Canonical UTC timestamp (ISO-8601 with tz)."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def _sha256_file(path: Path) -> str:
    """Streaming sha256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_physical_loc(source: str) -> int:
    """Count physical lines (canonical HNeRV parity L4 metric).

    Per HNeRV parity L4 the budget is "≤200 LOC". This is physical lines
    (including blanks + comments) because the contest scorer charges
    rate-term bytes on the bundled archive ONLY; the inflate.py LOC budget
    is a REVIEWABILITY metric (30-second-reviewable per the canonical
    discipline), so physical lines is the operator-facing metric.
    """
    if not source:
        return 0
    # Trailing newline is canonical per PEP 8; do not double-count.
    return len(source.splitlines())


def derive_submission_bundle_provenance(
    *,
    lane_id: str,
    substrate_id: str,
    archive_sha256: str,
    measurement_utc: str,
) -> dict[str, Any]:
    """Build the canonical Provenance dict for a submission bundle result.

    Per Catalog #323 canonical Provenance umbrella: every persisted row
    carries (axis_tag + evidence_grade + score_claim + promotable +
    canonical_helper_invocation + captured_at_utc).
    """
    return {
        "axis_tag": PREDICTED_AXIS_TAG,
        "evidence_grade": "[predicted; submission-bundle-canonical]",
        "score_claim": False,
        "promotable": False,
        "canonical_helper_invocation": (
            "tac.submission_packet.build_submission_bundle"
        ),
        "captured_at_utc": measurement_utc,
        "lane_id": lane_id,
        "substrate_id": substrate_id,
        "archive_sha256": archive_sha256,
        "canonical_equation_id": CANONICAL_EQUATION_ID,
        "canonical_equation_status": "FORMALIZATION_PENDING",
        "schema_version": SUBMISSION_BUNDLE_SCHEMA_VERSION,
    }


def _scan_for_local_absolute_paths(text: str) -> tuple[str, ...]:
    """Catalog #208: scan text for forbidden local-absolute-path patterns.

    Returns the canonical sorted tuple of matched substrings (empty when
    clean). Used by the linter sister at Layer 3 + by the bundle emitter
    to refuse README + report.txt content that would leak operator paths.
    """
    matches: list[str] = []
    for pattern in _LOCAL_ABSOLUTE_PATH_PATTERNS:
        for m in re.finditer(pattern, text):
            matches.append(m.group(0))
    return tuple(sorted(set(matches)))


def _scan_for_forbidden_pr_tokens(text: str) -> tuple[str, ...]:
    """Public-PR hygiene scan per CLAUDE.md + the forbidden-attribution
    standing directive.

    Returns the canonical sorted tuple of matched forbidden tokens (empty
    when clean). Used by the README emitter to refuse any draft that would
    leak Claude/Anthropic attribution into a public-PR surface.
    """
    matches: list[str] = []
    for token in _FORBIDDEN_PUBLIC_PR_TOKENS:
        if token in text:
            matches.append(token)
    return tuple(sorted(set(matches)))


def _emit_inflate_sh(
    *,
    output_dir: Path,
    archive_zip_name: str,
    python_invocation: str = "python3",
) -> Path:
    """Emit canonical Catalog #146 3-arg inflate.sh.

    Per Catalog #146 contest-compliant runtime template the inflate.sh
    signature is ``inflate.sh <archive_dir> <output_dir> <file_list>``;
    the script iterates each video name from file_list, computes the
    canonical .bin source path inside the archive dir, and invokes the
    bundled ``inflate.py`` per video.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "inflate.sh"
    contents = f"""#!/usr/bin/env bash
# Canonical inflate.sh per Catalog #146 3-arg signature.
# Emitted by tac.submission_packet.build_submission_bundle (Phase 4 Layer 2).
{CANONICAL_INFLATE_SH_REQUIRED_HEADER}

HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${{line%.*}}"
  SRC="${{DATA_DIR}}/x"
  if [ ! -f "$SRC" ]; then
    SRC="${{DATA_DIR}}/${{BASE}}.bin"
  fi
  DST="${{OUTPUT_DIR}}/${{BASE}}.raw"

  [ ! -f "$SRC" ] && echo "ERROR: ${{SRC}} not found" >&2 && exit 1

  printf "Inflating %s ... " "$line"
  "${{PYTHON:-{python_invocation}}}" "$HERE/inflate.py" "$SRC" "$DST"
done < "$FILE_LIST"
"""
    target.write_text(contents, encoding="utf-8")
    target.chmod(0o755)
    return target


def _emit_canonical_select_inflate_device_block() -> str:
    """Emit canonical Catalog #205 select_inflate_device byte-identical mirror.

    Per CLAUDE.md "Contest runtime closure" non-negotiable the inflate tree
    MUST be self-contained; that prevents an actual import of the
    canonical helper at inflate time. The mirror is byte-identical to
    ``tac.substrates._shared.inflate_runtime.select_inflate_device`` (modulo
    the ``torch.device`` return-type wrap on the substrate sister).
    """
    return '''def select_inflate_device():
    """Honor ``PACT_INFLATE_DEVICE`` (auto/cpu/cuda); MPS is forbidden.

    Canonical mirror of ``tac.substrates._shared.inflate_runtime.select_inflate_device``
    per Catalog #205. The mirror is required because CLAUDE.md "Contest
    runtime closure" forbids importing the canonical helper at inflate time
    (self-contained submission_dir invariant per HNeRV parity L4).
    """
    import os
    value = (os.environ.get("PACT_INFLATE_DEVICE") or "auto").strip().lower()
    if value == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if value == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("PACT_INFLATE_DEVICE=cuda but torch.cuda is not available")
        except ImportError as exc:
            raise RuntimeError("PACT_INFLATE_DEVICE=cuda but torch is unavailable") from exc
        return "cuda"
    if value == "cpu":
        return "cpu"
    raise RuntimeError(f"unsupported PACT_INFLATE_DEVICE={value!r}; expected auto/cpu/cuda")
'''


def _emit_inflate_py(
    *,
    output_dir: Path,
    substrate_id: str,
    archive_member_name: str,
    inflate_body: str | None = None,
    select_inflate_device_routing: str = SelectInflateDeviceRouting.INLINE_WITH_WAIVER.value,
) -> Path:
    """Emit canonical inflate.py per HNeRV parity L4.

    When ``inflate_body`` is provided, the substrate's bespoke decode logic
    is interpolated into the template (canonical for per-substrate
    distinguishing-feature decoders). When None, the template emits the
    canonical minimal scaffold for a NEW substrate scaffold per
    UNIQUE-AND-COMPLETE-PER-METHOD operating mode.

    Per HNeRV parity L4: the emitted body MUST be ≤200 LOC + ≤2 deps +
    numpy-portable + CUDA-or-CPU-agnostic + reviewable in 30s.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "inflate.py"

    select_block = _emit_canonical_select_inflate_device_block()
    inflate_default = (
        f'''    # Canonical scaffold inflate.py for substrate {substrate_id}.
    # Per HNeRV parity L4: this scaffold is ≤200 LOC + numpy-portable; the
    # substrate-specific decode logic is operator-routable via Phase 4
    # inflate_body= kwarg or via direct edit of this generated file.
    with open(src_bin, "rb") as f:
        archive_bytes = f.read()
    device = select_inflate_device()  # noqa: F841 — operator-routable
    # SUBSTRATE-SPECIFIC DECODE GOES HERE. The canonical scaffold writes a
    # placeholder zero-byte raw output so dispatch closure verifies; the
    # paired-auth-eval (Phase 6) is what surfaces empirical decode quality.
    with open(dst_raw, "wb") as fout:
        fout.write(b"")
'''
    )
    body = inflate_body if inflate_body is not None else inflate_default

    waiver_marker = ""
    if select_inflate_device_routing == SelectInflateDeviceRouting.INLINE_WITH_WAIVER.value:
        waiver_marker = "  # INLINE_DEVICE_FORK_OK:canonical_select_inflate_device_mirror_per_catalog_205_self_contained_inflate_runtime_per_hnerv_parity_L4"

    contents = f'''#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Canonical inflate.py emitted by tac.submission_packet.build_submission_bundle.

Phase 4 Layer 2 canonical scaffold for substrate {substrate_id!r}.

Per HNeRV parity L4: ≤200 LOC + ≤2 ext deps + numpy-portable + CUDA-or-CPU
agnostic + reviewable in 30s. Per Catalog #205: canonical select_inflate_device
routing (inline mirror with INLINE_DEVICE_FORK_OK waiver per the canonical
self-contained inflate-runtime contract).

Per Catalog #146: 3-arg inflate.sh contract delegates to this script per
video; this script reads ``<src.bin>`` and writes ``<dst.raw>``.

Per Catalog #295: PYTHONPATH self-containment — this scaffold has no bare
``from tac.*`` imports.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


{select_block}{waiver_marker}


def inflate(src_bin, dst_raw):
{body}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
'''
    target.write_text(contents, encoding="utf-8")
    return target


def _emit_readme(
    *,
    output_dir: Path,
    lane_id: str,
    substrate_id: str,
    archive_sha256: str,
    archive_bytes: int,
    attribution_chain_placeholder: str | None = None,
) -> Path:
    """Emit canonical README.md with attribution chain placeholder.

    Per PR 95 medal-class precedent + the standing directive
    `feedback_forbidden_claude_attribution_in_public_pr_surfaces.md`: the
    README MUST NOT mention Claude/Anthropic; attribution chain is placeholder
    for operator-trigger fill at Phase 7 + Phase 10.

    The README is internal-facing scaffold; the PR_BODY is emitted by Phase
    6/7 paired-auth-eval + attribution layers per the Phase 1 spec memo.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "README.md"

    attribution_section = (
        attribution_chain_placeholder
        if attribution_chain_placeholder is not None
        else (
            "<!-- Attribution chain placeholder; filled by Phase 6/7/10 layers. -->\n"
            "<!-- Per PR 95 medal-class precedent: @-mention chain + PR# hyperlink. -->"
        )
    )
    contents = f"""# Submission packet — {substrate_id}

Lane: `{lane_id}`
Substrate: `{substrate_id}`

## Archive identity (canonical regression surface)

| Field | Value |
|---|---|
| `archive_sha256` | `{archive_sha256}` |
| `archive_size_bytes` | `{archive_bytes:,}` |

## Runtime closure (HNeRV parity L4)

| File | Purpose |
|---|---|
| `inflate.sh` | Catalog #146 3-arg signature (archive_dir / output_dir / file_list) |
| `inflate.py` | Canonical numpy-portable decoder (≤200 LOC; CUDA-or-CPU agnostic) |
| `archive.zip` | Contest-compliant archive bytes |
| `archive_manifest.json` | Per-member identity manifest |
| `report.txt` | Placeholder; filled by Phase 6 paired-auth-eval |

## Attribution

{attribution_section}

## Reproducibility

- Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"
- Per CLAUDE.md "Apples-to-apples evidence discipline"
- Bundle emitted by `tac.submission_packet.build_submission_bundle` (Phase 4 Layer 2)
- Provenance per Catalog #323; observability-only per Catalog #341
"""

    # Catalog #208 + public-PR hygiene defense-in-depth at emit time.
    leaks = _scan_for_local_absolute_paths(contents)
    if leaks:
        raise SubmissionBundleError(
            f"README.md emit refused — Catalog #208 local-absolute-path leak: {leaks}"
        )
    forbidden = _scan_for_forbidden_pr_tokens(contents)
    if forbidden:
        raise SubmissionBundleError(
            f"README.md emit refused — public-PR hygiene leak: {forbidden}"
        )

    target.write_text(contents, encoding="utf-8")
    return target


def _emit_report_txt(
    *,
    output_dir: Path,
    archive_bytes: int,
) -> Path:
    """Emit canonical report.txt placeholder.

    Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": this is the
    canonical evaluation-report placeholder; Phase 6 paired-auth-eval
    OVERWRITES it with the actual contest evaluator output for the bundled
    archive on BOTH axes.

    The placeholder body mirrors the contest evaluator's actual output
    format so downstream Phase 5 compliance + Phase 6 paired_auth_eval
    consumers can validate the shape without re-emitting.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "report.txt"
    contents = f"""=== Evaluation config ===
  PLACEHOLDER — filled by Phase 6 paired_auth_eval per CLAUDE.md
  "Submission auth eval - BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
  HARDWARE" non-negotiable. Phase 4 bundle emitter writes only the
  canonical shape placeholder so downstream consumers can validate
  structurally.
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: PENDING
  Average SegNet Distortion: PENDING
  Submission file size: {archive_bytes:,} bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: PENDING
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = PENDING
"""
    target.write_text(contents, encoding="utf-8")
    return target


def _emit_archive_manifest_sidecar(
    *,
    output_dir: Path,
    archive_zip_path: Path,
    archive_sha256: str,
) -> Path:
    """Emit canonical archive_manifest.json per-member identity sidecar.

    Per PR101 / PR102 / PR103 medal-class precedent the archive_manifest
    carries per-member sha256 + size + CRC. The canonical helper at
    ``tac.submission_packet.archive_grammar`` emits the
    parser_section_manifest.json sister; THIS sidecar is the bundle-level
    archive identity surface.
    """
    import json
    import zipfile

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "archive_manifest.json"
    with zipfile.ZipFile(archive_zip_path) as zf:
        members = []
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            with zf.open(info.filename) as member:
                member_bytes = member.read()
                member_sha = hashlib.sha256(member_bytes).hexdigest()
            members.append(
                {
                    "name": info.filename,
                    "size": int(info.file_size),
                    "compressed_size": int(info.compress_size),
                    "crc32": int(info.CRC),
                    "sha256": member_sha,
                }
            )
    payload = {
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_zip_path.stat().st_size,
        "members": members,
        "member_count": len(members),
        "emitted_by": "tac.submission_packet.build_submission_bundle",
        "emitted_at_utc": _utc_now_iso(),
    }
    target.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def _copy_archive_zip(
    *,
    archive_grammar_manifest: ArchiveGrammarManifest,
    output_dir: Path,
    repo_root: Path,
) -> Path:
    """Copy the archive.zip from its source into the canonical
    submission_dir/archive.zip location.

    Per Catalog #361: when running on Modal workers the canonical
    ``output/submission/`` subtree MUST be mtime-fresh per the artifact
    harvester filter. This helper uses ``copy2`` then
    ``os.utime(dst, None)`` to set mtime to NOW (sister of canonical
    ``tac.substrates._shared.trainer_skeleton.vendor_module_with_fresh_mtime``).
    """
    import shutil

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "archive.zip"
    src_path = Path(archive_grammar_manifest.archive_path)
    src_abs = src_path if src_path.is_absolute() else (repo_root / src_path).resolve()
    if not src_abs.is_file():
        raise SubmissionBundleError(
            f"archive_grammar_manifest.archive_path {src_abs} does not exist"
        )
    shutil.copy2(src_abs, target)
    # Per Catalog #361: mtime-fresh so Modal harvester picks it up.
    os.utime(target, None)
    return target


def _verify_pythonpath_self_containment(
    inflate_py_path: Path,
    *,
    vendor_pythonpath_self_containment: bool,
) -> str:
    """Verify Catalog #295 PYTHONPATH self-containment for bundled inflate.py.

    Returns one of :class:`PythonpathSelfContainmentStatus` values.

    Per Catalog #295: the bundled inflate.py MUST NOT contain bare
    ``from tac.*`` imports without a vendored sister package alongside.
    The canonical scaffold this builder emits has zero ``from tac.*``
    imports so the default verdict is CLEAN.
    """
    source = inflate_py_path.read_text(encoding="utf-8")
    has_bare_tac_import = bool(
        re.search(r"^\s*(?:from\s+tac(?:\.|\s)|import\s+tac(?:\.|\s|$))", source, re.MULTILINE)
    )
    if not has_bare_tac_import:
        return PythonpathSelfContainmentStatus.CLEAN.value
    if vendor_pythonpath_self_containment:
        # Caller asserts the vendored sister package is present alongside;
        # this is operator-routable per Phase 5/6 dispatch-ready landing.
        submission_dir = inflate_py_path.parent
        vendored_tac_dir = submission_dir / "src" / "tac"
        vendored_tac_alongside = submission_dir / "tac"
        if vendored_tac_dir.is_dir() or vendored_tac_alongside.is_dir():
            return PythonpathSelfContainmentStatus.VENDORED_WITH_EXPLICIT_WAIVER.value
        raise SubmissionBundleError(
            "Catalog #295: inflate.py has bare `from tac.*` imports AND "
            "vendor_pythonpath_self_containment=True but no vendored "
            "submission_dir/src/tac/ or submission_dir/tac/ found"
        )
    raise SubmissionBundleError(
        "Catalog #295: inflate.py has bare `from tac.*` imports AND "
        "vendor_pythonpath_self_containment=False — submission would crash "
        "on a Modal worker per the empirical NSCS06 v5 anchor "
        "fc-01KRQMAQ7V41AFYMJH5HRK9P10"
    )


def _verify_inflate_sh_contest_compliant(inflate_sh_path: Path) -> None:
    """Verify Catalog #146 contest-compliant 3-arg signature.

    Per Catalog #146 the inflate.sh MUST honor the 3-arg signature
    (``archive_dir`` / ``output_dir`` / ``file_list``) and carry
    ``set -euo pipefail``. Raises :class:`SubmissionBundleError` on
    violation so the bundle-emit refuses to ship a non-compliant runtime.
    """
    source = inflate_sh_path.read_text(encoding="utf-8")
    if CANONICAL_INFLATE_SH_REQUIRED_HEADER not in source:
        raise SubmissionBundleError(
            f"Catalog #146: inflate.sh missing canonical header "
            f"{CANONICAL_INFLATE_SH_REQUIRED_HEADER!r}"
        )
    # Verify either positional ($1/$2/$3) OR named (DATA_DIR/OUTPUT_DIR/FILE_LIST) tokens present.
    has_positional = all(t in source for t in CANONICAL_INFLATE_SH_POSITIONAL_TOKENS)
    has_named = all(t in source for t in CANONICAL_INFLATE_SH_NAMED_TOKENS)
    if not (has_positional or has_named):
        raise SubmissionBundleError(
            "Catalog #146: inflate.sh missing 3-arg signature; "
            f"requires either {CANONICAL_INFLATE_SH_POSITIONAL_TOKENS} positional "
            f"OR {CANONICAL_INFLATE_SH_NAMED_TOKENS} named tokens"
        )


def build_dependency_closure_manifest(
    declared_dependencies: tuple[str, ...],
    *,
    dependency_budget: int = DEFAULT_INFLATE_DEPS_BUDGET,
    waiver_rationale: str | None = None,
) -> DependencyClosureManifest:
    """Build a canonical :class:`DependencyClosureManifest`.

    Args:
        declared_dependencies: full set of external deps the bundled
            inflate.py requires at runtime (canonical sorted on emit).
        dependency_budget: operator-set budget (canonical default
            :data:`DEFAULT_INFLATE_DEPS_BUDGET` per HNeRV parity L4).
        waiver_rationale: substantive rationale (≥4 chars,
            non-placeholder per Catalog #287) when over budget.

    Returns:
        Validated :class:`DependencyClosureManifest`.
    """
    sorted_deps = tuple(sorted(set(declared_dependencies)))
    within_budget = len(sorted_deps) <= dependency_budget
    numpy_portable = sorted_deps == ("numpy",)
    return DependencyClosureManifest(
        declared_dependencies=sorted_deps,
        dependency_budget=dependency_budget,
        within_budget=within_budget,
        numpy_portable=numpy_portable,
        waiver_rationale=waiver_rationale,
    )


def build_submission_bundle(
    *,
    compression_pipeline_result: CompressionPipelineResult,
    archive_grammar_manifest: ArchiveGrammarManifest,
    output_dir: Path,
    inflate_body: str | None = None,
    declared_dependencies: tuple[str, ...] = ("numpy",),
    inflate_py_loc_budget: int = DEFAULT_INFLATE_PY_LOC_BUDGET,
    inflate_deps_budget: int = DEFAULT_INFLATE_DEPS_BUDGET,
    inflate_py_loc_waiver_rationale: str | None = None,
    inflate_deps_waiver_rationale: str | None = None,
    vendor_pythonpath_self_containment: bool = True,
    select_inflate_device_routing: str = (
        SelectInflateDeviceRouting.INLINE_WITH_WAIVER.value
    ),
    attribution_chain_placeholder: str | None = None,
    python_invocation: str = "python3",
    repo_root: Path | None = None,
) -> SubmissionBundleResult:
    """Canonical submission bundle builder (Layer 2) — main entry point.

    Builds ``submission_dir/`` from the Phase 2 + Phase 3 lineage:
    inflate.sh per Catalog #146 + inflate.py per HNeRV parity L4 + Catalog
    #205 + Catalog #295 + archive.zip per Catalog #361 mtime-fresh +
    README.md per PR 95 precedent + report.txt placeholder per Phase 6 +
    archive_manifest.json per-member identity sidecar.

    The helper does NOT invoke paid Modal / Vast.ai / Lightning dispatch
    per Phase 4 scope. It DERIVES the canonical bundle from existing
    archive bytes (Phase 3 ArchiveGrammarManifest) + Phase 2 compression
    pipeline lineage. Phase 6 ``paired_auth_eval`` is where paired-axis
    empirical anchors land + REWRITES the placeholder report.txt with
    real evaluator output.

    Args:
        compression_pipeline_result: typed Phase 2 Layer 0 result whose
            ``lane_id`` + ``substrate_id`` lineage is canonical.
        archive_grammar_manifest: typed Phase 3 Layer 1 manifest whose
            ``archive_sha256`` + ``archive_path`` lineage is canonical.
        output_dir: directory to bundle into (canonical:
            ``submissions/pr<N>_<lane>/`` or
            ``experiments/results/<lane>/submission_dir/``).
        inflate_body: optional substrate-specific decode logic body (
            indented 4 spaces; interpolated into the canonical scaffold).
            When None, the canonical scaffold emits a zero-byte placeholder
            decode so dispatch closure can verify structurally.
        declared_dependencies: canonical sorted tuple of external deps the
            bundled inflate.py requires at runtime. Canonical numpy-portable
            default is ``("numpy",)``; HNeRV-class default is ``("numpy", "torch")``.
        inflate_py_loc_budget: HNeRV parity L4 LOC budget (canonical
            default :data:`DEFAULT_INFLATE_PY_LOC_BUDGET`).
        inflate_deps_budget: HNeRV parity L4 deps budget (canonical
            default :data:`DEFAULT_INFLATE_DEPS_BUDGET`).
        inflate_py_loc_waiver_rationale: substantive rationale when LOC
            exceeds budget (≥4 chars per Catalog #287).
        inflate_deps_waiver_rationale: substantive rationale when deps
            exceed budget (≥4 chars per Catalog #287).
        vendor_pythonpath_self_containment: when True (canonical default
            per Catalog #295), the helper verifies inflate.py has no bare
            ``from tac.*`` imports OR that a vendored sister package is
            present alongside.
        select_inflate_device_routing: one of :class:`SelectInflateDeviceRouting`
            values per Catalog #205.
        attribution_chain_placeholder: optional pre-built attribution chain
            markdown (Phase 6 attribution layer fills this).
        python_invocation: canonical python invocation for inflate.sh
            (default ``"python3"`` per contest reference).
        repo_root: override repo root (defaults to module-resolved REPO_ROOT).

    Returns:
        :class:`SubmissionBundleResult` with canonical Provenance per
        Catalog #323.

    Raises:
        SubmissionBundleError: when HNeRV parity L4 invariants violated
            AND no waiver supplied OR Catalog #146/#205/#295 invariants
            violated.
    """
    started = _utc_now_iso()
    started_perf = datetime.datetime.now(datetime.UTC)
    root = repo_root if repo_root is not None else REPO_ROOT

    if not isinstance(compression_pipeline_result, CompressionPipelineResult):
        raise SubmissionBundleError(
            "compression_pipeline_result must be a "
            "tac.submission_packet.compression_pipeline.CompressionPipelineResult"
        )
    if not isinstance(archive_grammar_manifest, ArchiveGrammarManifest):
        raise SubmissionBundleError(
            "archive_grammar_manifest must be a "
            "tac.submission_packet.archive_grammar.ArchiveGrammarManifest"
        )
    if compression_pipeline_result.lane_id != archive_grammar_manifest.lane_id:
        raise SubmissionBundleError(
            f"lane_id mismatch: compression_pipeline.lane_id="
            f"{compression_pipeline_result.lane_id!r} vs "
            f"archive_grammar.lane_id={archive_grammar_manifest.lane_id!r}"
        )
    if compression_pipeline_result.substrate_id != archive_grammar_manifest.substrate_id:
        raise SubmissionBundleError(
            f"substrate_id mismatch: compression_pipeline="
            f"{compression_pipeline_result.substrate_id!r} vs "
            f"archive_grammar={archive_grammar_manifest.substrate_id!r}"
        )
    if not isinstance(output_dir, Path):
        raise SubmissionBundleError("output_dir must be a pathlib.Path")

    output_dir = (
        output_dir if output_dir.is_absolute() else (root / output_dir).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: copy archive.zip into output_dir per Catalog #361 mtime-fresh.
    archive_zip_path = _copy_archive_zip(
        archive_grammar_manifest=archive_grammar_manifest,
        output_dir=output_dir,
        repo_root=root,
    )
    archive_sha = _sha256_file(archive_zip_path)
    archive_size = archive_zip_path.stat().st_size
    if archive_sha != archive_grammar_manifest.archive_sha256:
        raise SubmissionBundleError(
            f"archive_sha256 mismatch after copy: "
            f"manifest={archive_grammar_manifest.archive_sha256[:12]}..., "
            f"copied={archive_sha[:12]}..."
        )

    # Step 2: emit inflate.sh per Catalog #146 + verify contest-compliance.
    inflate_sh_path = _emit_inflate_sh(
        output_dir=output_dir,
        archive_zip_name=archive_zip_path.name,
        python_invocation=python_invocation,
    )
    _verify_inflate_sh_contest_compliant(inflate_sh_path)

    # Step 3: emit inflate.py per HNeRV parity L4 + Catalog #205.
    inflate_py_path = _emit_inflate_py(
        output_dir=output_dir,
        substrate_id=compression_pipeline_result.substrate_id,
        archive_member_name=archive_grammar_manifest.section_specs[0].member_name,
        inflate_body=inflate_body,
        select_inflate_device_routing=select_inflate_device_routing,
    )
    inflate_source = inflate_py_path.read_text(encoding="utf-8")
    inflate_py_loc = _count_physical_loc(inflate_source)
    if inflate_py_loc > inflate_py_loc_budget:
        if inflate_py_loc_waiver_rationale is None:
            raise SubmissionBundleError(
                f"HNeRV parity L4: inflate.py LOC={inflate_py_loc} exceeds budget "
                f"{inflate_py_loc_budget}; non-None inflate_py_loc_waiver_rationale required"
            )

    # Step 4: verify Catalog #295 PYTHONPATH self-containment.
    pythonpath_status = _verify_pythonpath_self_containment(
        inflate_py_path,
        vendor_pythonpath_self_containment=vendor_pythonpath_self_containment,
    )

    # Step 5: build dependency closure manifest per HNeRV parity L4.
    dep_manifest = build_dependency_closure_manifest(
        declared_dependencies=declared_dependencies,
        dependency_budget=inflate_deps_budget,
        waiver_rationale=inflate_deps_waiver_rationale,
    )

    # Step 6: emit README.md per PR 95 precedent + Catalog #208 + public-PR hygiene.
    readme_path = _emit_readme(
        output_dir=output_dir,
        lane_id=compression_pipeline_result.lane_id,
        substrate_id=compression_pipeline_result.substrate_id,
        archive_sha256=archive_sha,
        archive_bytes=archive_size,
        attribution_chain_placeholder=attribution_chain_placeholder,
    )

    # Step 7: emit report.txt placeholder per Phase 6 paired-auth-eval.
    report_path = _emit_report_txt(
        output_dir=output_dir,
        archive_bytes=archive_size,
    )

    # Step 8: emit archive_manifest.json sidecar per PR101/PR102/PR103 precedent.
    archive_manifest_path = _emit_archive_manifest_sidecar(
        output_dir=output_dir,
        archive_zip_path=archive_zip_path,
        archive_sha256=archive_sha,
    )

    measurement_utc = _utc_now_iso()
    canonical_provenance = derive_submission_bundle_provenance(
        lane_id=compression_pipeline_result.lane_id,
        substrate_id=compression_pipeline_result.substrate_id,
        archive_sha256=archive_sha,
        measurement_utc=measurement_utc,
    )

    elapsed = (datetime.datetime.now(datetime.UTC) - started_perf).total_seconds()

    result = SubmissionBundleResult(
        schema_version=SUBMISSION_BUNDLE_SCHEMA_VERSION,
        lane_id=compression_pipeline_result.lane_id,
        substrate_id=compression_pipeline_result.substrate_id,
        archive_sha256=archive_sha,
        archive_bytes=int(archive_size),
        submission_dir=str(output_dir),
        inflate_sh_path=str(inflate_sh_path),
        inflate_py_path=str(inflate_py_path),
        inflate_py_loc=int(inflate_py_loc),
        inflate_py_loc_budget=int(inflate_py_loc_budget),
        inflate_py_loc_waiver_rationale=inflate_py_loc_waiver_rationale,
        readme_md_path=str(readme_path),
        report_txt_path=str(report_path),
        archive_manifest_path=str(archive_manifest_path),
        dependency_closure_manifest=dep_manifest,
        select_inflate_device_routing=select_inflate_device_routing,
        pythonpath_self_containment_status=pythonpath_status,
        vendor_pythonpath_self_containment=bool(vendor_pythonpath_self_containment),
        runtime_dep_closure=dep_manifest.declared_dependencies,
        measurement_utc=measurement_utc,
        axis_tag=PREDICTED_AXIS_TAG,
        score_claim=False,
        promotable=False,
        evidence_grade="[predicted; submission-bundle-canonical]",
        canonical_helper_invocation="tac.submission_packet.build_submission_bundle",
        canonical_equation_id=CANONICAL_EQUATION_ID,
        canonical_equation_status="FORMALIZATION_PENDING",
        elapsed_seconds=float(elapsed),
        canonical_provenance=canonical_provenance,
        written_at_utc=measurement_utc,
        written_pid=os.getpid(),
        written_host=socket.gethostname(),
    )
    return result
