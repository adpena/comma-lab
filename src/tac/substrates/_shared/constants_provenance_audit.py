# SPDX-License-Identifier: MIT
"""Canonical detector for the ARBITRARINESS bug class (Catalog #385).

Source: operator hardening packet 2026-06-09. This is the reusable detection logic
for ``tac.preflight.check_no_arbitrary_score_relevant_constant_at_l2``; the thin
preflight wrapper delegates here. Per AGENTS.md "TAC / comma-lab Boundary": the
real reusable detection logic lives in ``tac``; ``preflight.py`` carries only the
thin orchestrator wrapper.

The rule (a falling-rule list, per CLAUDE.md "Preflight failure messages must cite
the rule chain"): scan every ``.omx/state/constants_provenance/*.json`` manifest.
A manifest VIOLATES when its ``ConstantsProvenanceManifest.blocking_findings()`` is
non-empty — i.e. the vehicle declares maturity >= a constant's
``blocking_maturity_level`` (default L2 = intrinsically optimized) while that
constant is (``score_relevant`` OR ``stability_critical``) AND tagged ``ARBITRARY``
AND has NO real ``replacement_path``.

GUARDRAIL (operator-explicit): only ``score_relevant`` OR ``stability_critical``
constants block — harmless engineering constants are exempt by construction (the
manifest's :meth:`ConstantProvenance.is_gated` filter). This detector inherits that
guardrail; it does not re-implement the blocking logic (single source of truth is
the manifest's own ``blocks_at``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .constants_provenance_manifest import (
    ConstantsProvenanceManifest,
    constants_provenance_state_dir,
)

__all__ = [
    "ConstantsProvenanceFinding",
    "audit_constants_provenance_manifests",
]


@dataclass(frozen=True)
class ConstantsProvenanceFinding:
    """One blocking finding from a vehicle's constants-provenance manifest."""

    vehicle_id: str
    manifest_path: str
    declared_maturity_level: str
    finding: str

    def message(self) -> str:
        return (
            f"[{self.vehicle_id} @ {self.declared_maturity_level}] "
            f"({self.manifest_path}): {self.finding}"
        )


def audit_constants_provenance_manifests(
    repo_root: str | Path | None = None,
    *,
    manifest_dir: str | Path | None = None,
    include_canonical: bool = True,
) -> list[ConstantsProvenanceFinding]:
    """Return blocking findings across all constants-provenance manifests.

    Two manifest sources are merged (the canonical module is the source-of-truth;
    the durable JSON directory is the operator-override surface):

    1. **Canonical module** — ``CANONICAL_CONSTANTS_MANIFESTS`` from
       ``constants_provenance_manifests_canonical`` (the committed seed; works on a
       fresh checkout where no JSON has been emitted). Skipped when
       ``include_canonical=False`` (the test path passes its own isolated dir).
    2. **Durable JSON directory** — ``manifest_dir`` (default
       ``.omx/state/constants_provenance``), gitignored per the
       ``.omx/state/*`` rule (a regenerable emission of the canonical module +
       any operator-emitted per-vehicle overrides). A JSON manifest OVERRIDES the
       canonical module entry for the same ``vehicle_id`` (latest-emit-wins), so an
       operator can advance a vehicle's declared maturity / resolve a constant
       without editing the seed module.

    Each manifest's ``blocking_findings()`` entries are collected. A malformed
    JSON (bad schema / missing field) is itself surfaced as a finding (fail-closed:
    a manifest the gate cannot parse must not silently pass).

    Returns an empty list when every manifest is clean (no declared-L2+ vehicle has
    an unresolved ARBITRARY score-relevant/stability-critical constant).
    """
    # vehicle_id -> (manifest, source_label) ; JSON overrides canonical.
    merged: dict[str, tuple[ConstantsProvenanceManifest, str]] = {}
    parse_findings: list[ConstantsProvenanceFinding] = []

    if include_canonical:
        # Local import to avoid a module-load cycle (canonical imports the schema).
        from .constants_provenance_manifests_canonical import (
            CANONICAL_CONSTANTS_MANIFESTS,
        )

        for manifest in CANONICAL_CONSTANTS_MANIFESTS:
            merged[manifest.vehicle_id] = (manifest, "canonical_module")

    scan_dir = (
        Path(manifest_dir)
        if manifest_dir is not None
        else constants_provenance_state_dir(repo_root)
    )
    if scan_dir.exists():
        for path in sorted(scan_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                parse_findings.append(
                    ConstantsProvenanceFinding(
                        vehicle_id=path.stem,
                        manifest_path=str(path),
                        declared_maturity_level="?",
                        finding=f"UNPARSEABLE manifest (fail-closed): {exc}",
                    )
                )
                continue
            try:
                manifest = ConstantsProvenanceManifest.from_dict(payload)
            except (KeyError, ValueError, TypeError) as exc:
                parse_findings.append(
                    ConstantsProvenanceFinding(
                        vehicle_id=str(payload.get("vehicle_id", path.stem)),
                        manifest_path=str(path),
                        declared_maturity_level=str(
                            payload.get("declared_maturity_level", "?")
                        ),
                        finding=f"MALFORMED manifest (fail-closed): {exc}",
                    )
                )
                continue
            merged[manifest.vehicle_id] = (manifest, str(path))

    findings: list[ConstantsProvenanceFinding] = list(parse_findings)
    for vehicle_id in sorted(merged):
        manifest, source = merged[vehicle_id]
        for finding in manifest.blocking_findings():
            findings.append(
                ConstantsProvenanceFinding(
                    vehicle_id=manifest.vehicle_id,
                    manifest_path=source,
                    declared_maturity_level=manifest.declared_maturity_level,
                    finding=finding,
                )
            )
    return findings
