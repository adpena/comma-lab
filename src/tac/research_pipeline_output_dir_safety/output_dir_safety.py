# SPDX-License-Identifier: MIT
"""Canonical output-dir safety validator for ``.omx/research/`` writers.

See package ``__init__`` docstring for the canonical contract + cross-
references.

The 4-cascade falling-rule validator:

1. **CASCADE A — out of scope**: ``output_dir`` is NOT under
   ``.omx/research/`` -> PROCEED. The gate is scoped to the HISTORICAL_PROVENANCE
   namespace.

2. **CASCADE B — fresh dir**: ``output_dir`` does NOT exist OR is empty
   (no HISTORICAL_PROVENANCE JSON files) -> PROCEED. A first-write to a
   fresh dir is canonical per-invocation partition.

3. **CASCADE C — explicit operator opt-in**: caller passed
   ``allow_overwrite_existing_historical_provenance=True`` AND a
   ``waiver_rationale`` non-placeholder (>=4 chars; placeholder
   ``<rationale>``/``<reason>``/empty rejected per Catalog #287) ->
   PROCEED (with explicit opt-in marker).

4. **CASCADE D — refuse**: ``output_dir`` contains HISTORICAL_PROVENANCE
   JSON files AND no opt-in -> REFUSE with diagnostic verdict.

The validator is a pure function with no side effects (it does not write,
delete, or move any files). Callers route their CLI ``--output-dir``
through this validator immediately after ``argparse`` so the fail-closed
decision happens BEFORE any expensive computation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

# Canonical regex patterns identifying dated-subdir naming conventions.
# A dir whose basename matches any of these is presumed per-invocation safe.
DEFAULT_DATED_SUFFIX_PATTERNS: tuple[str, ...] = (
    r".*_\d{8}T\d{6}Z$",  # canonical _YYYYMMDDTHHMMSSZ
    r".*_\d{8}T\d{4}Z$",  # _YYYYMMDDTHHMMZ
    r".*_\d{8}Z$",  # _YYYYMMDDZ
    r".*_\d{8}$",  # _YYYYMMDD
    r".*_\d{14}$",  # _YYYYMMDDHHMMSS
)

# Placeholder rationale tokens forbidden per Catalog #287 sister discipline so
# the canonical helper docstring example cannot self-waive.
DEFAULT_WAIVER_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "<rationale>",
        "<reason>",
        "<reason-here>",
        "<rationale-here>",
        "<rationale_here>",
        "<reason_here>",
        "tbd",
        "todo",
        "fixme",
        "xxx",
        "placeholder",
    }
)

# Canonical HISTORICAL_PROVENANCE JSON file basenames that, if present in
# the target output_dir, indicate a prior invocation already wrote
# canonical evidence. Re-running with overwrite would mutate them.
#
# This is a defensive minimum set covering the 3 anchor cases from the
# 2026-05-28T22:41 audit (pr95_mlx_runtime_consumption_queue /
# repair_multi_archive_autonomous_live_psv3_fec6 /
# frontier_final_rate_attack_fp11_brotli_exec3); callers can extend via
# the ``additional_canonical_filenames`` parameter.
HISTORICAL_PROVENANCE_JSON_NAMES: frozenset[str] = frozenset(
    {
        "manifest.json",
        "runner_summary.json",
        "run_summary.json",
        "plan.json",
        "queue.json",
        "queue_observe.json",
        "queue_performance.json",
        "queue_validate.json",
        "score_report.json",
        "work_order.json",
        "archive_discovery.json",
        "submission_runtime_closure_report.json",
        "runtime_consumption_proof.json",
        "representation_training_manifest.json",
        "pr95_public_archive_export.json",
        "mlx_gpu_decoder_trace.json",
        "mlx_gpu_forward_drift_attestation.json",
        "execution_report.json",
        "post_execute_feedback_refresh.json",
        "bootstrap.json",
        "experiment_queue.json",
        "materializer_contexts.json",
        "target_coverage.json",
        "materializer_backlog.json",
        "materializer_work_queue.json",
        "derived_section_manifests.json",
        "signal_harvest.json",
    }
)


class OutputDirSafetyError(RuntimeError):
    """Raised when fail-closed enforcement refuses an output dir."""


@dataclass(frozen=True)
class ResearchPipelineOutputDirVerdict:
    """Frozen verdict for a single ``output_dir`` validation."""

    output_dir: str
    recommendation: str  # "PROCEED" | "REFUSE_EXISTING_HISTORICAL_PROVENANCE"
    cascade_matched: str  # "A_OUT_OF_SCOPE" | "B_FRESH" | "C_EXPLICIT_OPT_IN" | "D_REFUSE"
    is_under_omx_research: bool
    existing_canonical_json_files: tuple[str, ...]
    dated_suffix_matched: bool
    waiver_rationale_accepted: bool
    waiver_rationale_rejection_reason: str
    diagnostic_message: str
    operator_routable_unwind_path: str
    canonical_anti_pattern_id: str = (
        "research_pipeline_tool_re_writes_historical_provenance_json_with_mutated_fields_v1"
    )


def _is_under_omx_research(output_dir: Path, repo_root: Path) -> bool:
    """Return True iff ``output_dir`` is under ``<repo_root>/.omx/research/``."""

    try:
        rel = output_dir.resolve(strict=False).relative_to(
            (repo_root / ".omx" / "research").resolve(strict=False)
        )
    except (ValueError, OSError):
        return False
    # Reject the namespace root itself (only subdirs are canonical write
    # targets per the canonical helper contract).
    return rel.parts and rel.parts[0] != ".."


def is_dated_subdir(
    output_dir: Path,
    *,
    patterns: tuple[str, ...] = DEFAULT_DATED_SUFFIX_PATTERNS,
) -> bool:
    """Return True iff ``output_dir.name`` matches any canonical dated pattern."""

    name = output_dir.name
    for pat in patterns:
        if re.match(pat, name):
            return True
    return False


def _list_existing_canonical_jsons(
    output_dir: Path,
    *,
    additional_canonical_filenames: frozenset[str],
) -> tuple[str, ...]:
    """Return the names of HISTORICAL_PROVENANCE JSON files already in ``output_dir``."""

    if not output_dir.is_dir():
        return ()
    canonical = HISTORICAL_PROVENANCE_JSON_NAMES | additional_canonical_filenames
    found: list[str] = []
    try:
        for child in output_dir.iterdir():
            if child.is_file() and child.name in canonical:
                found.append(child.name)
    except OSError:
        return ()
    return tuple(sorted(found))


def _waiver_rationale_accepted(
    rationale: str | None,
) -> tuple[bool, str]:
    """Per Catalog #287: rationale must be non-placeholder ≥4 chars."""

    if rationale is None:
        return False, "missing rationale"
    stripped = rationale.strip()
    if not stripped:
        return False, "empty rationale"
    if len(stripped) < 4:
        return False, f"rationale too short ({len(stripped)} chars; min 4)"
    if stripped.lower() in DEFAULT_WAIVER_PLACEHOLDERS:
        return False, f"placeholder rationale rejected ({stripped!r})"
    return True, ""


def validate_research_pipeline_output_dir(
    output_dir: str | Path,
    *,
    repo_root: str | Path,
    allow_overwrite_existing_historical_provenance: bool = False,
    waiver_rationale: str | None = None,
    additional_canonical_filenames: frozenset[str] | Mapping[str, object] | None = None,
) -> ResearchPipelineOutputDirVerdict:
    """4-cascade falling-rule validator. See module docstring."""

    out_path = Path(output_dir)
    repo = Path(repo_root)
    under_research = _is_under_omx_research(out_path, repo)
    dated_match = is_dated_subdir(out_path)

    extra_filenames: frozenset[str]
    if additional_canonical_filenames is None:
        extra_filenames = frozenset()
    elif isinstance(additional_canonical_filenames, frozenset):
        extra_filenames = additional_canonical_filenames
    else:
        extra_filenames = frozenset(map(str, additional_canonical_filenames))

    existing = _list_existing_canonical_jsons(
        out_path, additional_canonical_filenames=extra_filenames
    )

    waiver_ok, waiver_reject = _waiver_rationale_accepted(waiver_rationale)

    # Cascade A — out of scope
    if not under_research:
        return ResearchPipelineOutputDirVerdict(
            output_dir=str(out_path),
            recommendation="PROCEED",
            cascade_matched="A_OUT_OF_SCOPE",
            is_under_omx_research=False,
            existing_canonical_json_files=existing,
            dated_suffix_matched=dated_match,
            waiver_rationale_accepted=waiver_ok,
            waiver_rationale_rejection_reason=waiver_reject,
            diagnostic_message=(
                f"output_dir {str(out_path)!r} not under <repo>/.omx/research/ "
                f"-- HISTORICAL_PROVENANCE namespace gate does not apply"
            ),
            operator_routable_unwind_path="",
        )

    # Cascade B — fresh dir (does not exist, or exists with no canonical JSONs)
    if not existing:
        return ResearchPipelineOutputDirVerdict(
            output_dir=str(out_path),
            recommendation="PROCEED",
            cascade_matched="B_FRESH",
            is_under_omx_research=True,
            existing_canonical_json_files=(),
            dated_suffix_matched=dated_match,
            waiver_rationale_accepted=waiver_ok,
            waiver_rationale_rejection_reason=waiver_reject,
            diagnostic_message=(
                f"output_dir {str(out_path)!r} is a fresh per-invocation partition "
                f"(no canonical HISTORICAL_PROVENANCE JSONs present)"
            ),
            operator_routable_unwind_path="",
        )

    # Cascade C — explicit operator opt-in
    if allow_overwrite_existing_historical_provenance and waiver_ok:
        return ResearchPipelineOutputDirVerdict(
            output_dir=str(out_path),
            recommendation="PROCEED",
            cascade_matched="C_EXPLICIT_OPT_IN",
            is_under_omx_research=True,
            existing_canonical_json_files=existing,
            dated_suffix_matched=dated_match,
            waiver_rationale_accepted=True,
            waiver_rationale_rejection_reason="",
            diagnostic_message=(
                f"output_dir {str(out_path)!r} contains "
                f"{len(existing)} canonical HISTORICAL_PROVENANCE JSON file(s) "
                f"but operator explicitly opted in via waiver rationale: "
                f"{waiver_rationale!r}"
            ),
            operator_routable_unwind_path="",
        )

    # Cascade D — refuse
    unwind = (
        "Refusing to write canonical HISTORICAL_PROVENANCE JSONs to an output_dir "
        "that already contains them (per Catalog #113 artifact_lifecycle "
        "discipline + canonical anti-pattern "
        "research_pipeline_tool_re_writes_historical_provenance_json_with_mutated_fields_v1). "
        "Canonical unwind path: (a) re-run with a fresh dated-subdir per invocation "
        "(e.g. <basename>_YYYYMMDDTHHMMSSZ); (b) re-classify the target as "
        "DERIVED_OUTPUT in .omx/state/artifact_kind_registry.yaml and add a "
        "regeneration header per Catalog #113 DerivedOutputGuard; (c) pass "
        "allow_overwrite_existing_historical_provenance=True with a substantive "
        "waiver_rationale (>=4 chars, non-placeholder per Catalog #287) "
        "to opt-in explicitly."
    )

    return ResearchPipelineOutputDirVerdict(
        output_dir=str(out_path),
        recommendation="REFUSE_EXISTING_HISTORICAL_PROVENANCE",
        cascade_matched="D_REFUSE",
        is_under_omx_research=True,
        existing_canonical_json_files=existing,
        dated_suffix_matched=dated_match,
        waiver_rationale_accepted=waiver_ok,
        waiver_rationale_rejection_reason=waiver_reject,
        diagnostic_message=(
            f"output_dir {str(out_path)!r} already contains "
            f"{len(existing)} canonical HISTORICAL_PROVENANCE JSON file(s): "
            f"{', '.join(existing)}"
        ),
        operator_routable_unwind_path=unwind,
    )


def enforce_research_pipeline_output_dir(
    output_dir: str | Path,
    *,
    repo_root: str | Path,
    allow_overwrite_existing_historical_provenance: bool = False,
    waiver_rationale: str | None = None,
    additional_canonical_filenames: frozenset[str] | Mapping[str, object] | None = None,
) -> ResearchPipelineOutputDirVerdict:
    """Convenience wrapper: validate and raise on non-PROCEED verdict."""

    verdict = validate_research_pipeline_output_dir(
        output_dir,
        repo_root=repo_root,
        allow_overwrite_existing_historical_provenance=(
            allow_overwrite_existing_historical_provenance
        ),
        waiver_rationale=waiver_rationale,
        additional_canonical_filenames=additional_canonical_filenames,
    )
    if verdict.recommendation != "PROCEED":
        raise OutputDirSafetyError(
            f"{verdict.diagnostic_message}\n{verdict.operator_routable_unwind_path}"
        )
    return verdict
