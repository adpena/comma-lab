# SPDX-License-Identifier: MIT
"""The Vehicle-OS compiler dashboard generator.

Source: operator binding directive 2026-06-09 (the fleet-wide-meta-bug crux) +
``docs/vehicle_operating_system.md`` "Dashboard discipline": *"A generated
dashboard (`pact_compiler_dashboard.{json,md}`) lists, per vehicle: maturity
level, current ALLOWED claim, latest artifact, authority_tier, metric_family,
current blocker, next command, owner, pass route, fail route. Work is
dashboard-driven; no stale-memory decisions."*

This module is the reusable, fail-soft renderer. The thin CLI
``tools/render_pact_compiler_dashboard.py`` delegates here per AGENTS.md
"tac stays clean; comma-lab owns research state" (research-state custody +
operator-facing dashboards live in ``src/comma_lab/``, not ``tac``).

Design discipline (per CLAUDE.md):

* **Pointer-only scores** ("Frontier scores are pointer-only"): the frontier
  section is read live from ``.omx/state/canonical_frontier_pointer.json`` —
  NEVER hardcoded. A missing pointer yields an explicit ``POINTER_MISSING``
  row, not a fabricated score.
* **Fail-soft with explicit AUDIT_PENDING rows**: a missing manifest does not
  crash the generator; the vehicle's row is emitted with the honest
  ``AUDIT_PENDING`` sentinel and a ``manifest_missing`` blocker so the operator
  sees the gap rather than nothing (Catalog #287 placeholder discipline:
  AUDIT_PENDING is a real status).
* **Maturity FROM EVIDENCE**: every assigned ``maturity_level`` cites the
  artifact / manifest condition that justifies it (the ``maturity_evidence``
  field). No level is asserted without a machine-readable basis.
* **Fresh-checkout safe**: scores come from the committed pointer; manifests
  from ``.omx/state`` (regenerable) AND the canonical seed module where one
  exists. Optional VertigoDataTier verdict JSONs are read when present and
  degrade to AUDIT_PENDING when the SSD tier is not mounted.

The L0-L7 ladder + allowed-claim semantics are canonical in
``docs/vehicle_operating_system.md``; this module encodes them as data so the
dashboard and the OS doc never drift.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from tac.substrates._shared.objective_reachability_manifest import (
    ObjectiveReachabilityManifest,
    objective_reachability_path_for_vehicle,
)
from tac.substrates._shared.vehicle_fidelity_manifest import (
    VehicleFidelityManifest,
    manifest_path_for_vehicle,
)

__all__ = [
    "AUDIT_PENDING",
    "MATURITY_LADDER",
    "VEHICLE_SPECS",
    "DashboardModel",
    "VehicleRow",
    "build_dashboard_model",
    "default_repo_root",
    "render_json",
    "render_markdown",
    "ssd_tier_roots",
    "write_dashboard",
]

# Honest not-yet-known sentinel (Catalog #287: a real status, not "TBD").
AUDIT_PENDING = "AUDIT_PENDING"

# The canonical L0-L7 ladder with the OS doc's allowed-claim per level.
# (docs/vehicle_operating_system.md "The maturity ladder (L0-L7)".)
MATURITY_LADDER: tuple[tuple[str, str, str], ...] = (
    ("L0", "Sketch", "research_carrier_sketch"),
    ("L1", "Mechanism-present", "mechanism_present_unit_tested"),
    ("L2", "Intrinsically optimized", "intrinsic_native_task_passed"),
    ("L3", "Archive-real", "archive_real_byte_closed_consumed"),
    ("L4", "Exact-scored", "exact_scored_row_exists"),
    ("L5", "Contextually optimized", "v3_delta_s_compared"),
    ("L6", "Composable", "commutator_measured"),
    ("L7", "Promotion-ready", "paired_cpu_cuda_same_archive"),
)

_MATURITY_NAME: Mapping[str, str] = {lvl: name for lvl, name, _ in MATURITY_LADDER}
_MATURITY_CLAIM: Mapping[str, str] = {lvl: claim for lvl, _, claim in MATURITY_LADDER}


@dataclass(frozen=True)
class VehicleSpec:
    """Static per-vehicle wiring (the manifest ids + owner + routes).

    ``fidelity_id`` and ``reachability_id`` differ for SNeRV (the fidelity
    manifest keys on ``snerv_inverse_steg_carrier`` while reachability keys on
    ``snerv``). ``constants_id`` is the constants-provenance manifest id where
    one exists (the canonical place the ``declared_maturity_level`` lives), or
    ``None`` when no constants manifest is emitted for the vehicle.
    """

    vehicle: str
    """Canonical dashboard vehicle key (e.g. ``snerv``, ``hi_nerv``)."""

    family: str
    """The literature family the vehicle claims (for operator context)."""

    fidelity_id: str | None
    reachability_id: str | None
    constants_id: str | None
    owner: str
    pass_route: str
    fail_route: str
    is_vehicle: bool = True
    """False for infrastructure (atlas / V3) — rendered as ``n/a-vehicle``."""

    verdict_glob: tuple[str, ...] = ()
    """Optional SSD-tier glob(s) for the latest typed verdict JSON."""


# The fleet, in OS-doc order (V0..V5 + infrastructure). The strategic memo
# `.omx/research/strategic_reevaluation_weeks_to_centuries_20260609.md` is the
# measured-fleet source; this list binds each vehicle to its manifest ids.
VEHICLE_SPECS: tuple[VehicleSpec, ...] = (
    VehicleSpec(
        vehicle="snerv",
        family="SNeRV (Spectra-preserving NeRV, arXiv 2501.01681)",
        fidelity_id="snerv_inverse_steg_carrier",
        reachability_id="snerv",
        constants_id=None,
        owner="snerv_branch_b_rate_attack",
        pass_route="LF entropy-coding front lowers rate -> V3 judges first real program -> V5 atoms stack",
        fail_route="store-LF rate chasm is the research front (entropy-code the LF; SNeRV store-LF/generate-HF)",
        verdict_glob=(
            "snerv_mistake_b_g1a_*/snerv_g1b_export_binding_verdict.v1.json",
        ),
    ),
    VehicleSpec(
        vehicle="hi_nerv",
        family="HiNeRV (Hierarchical NeRV, arXiv 2306.09818)",
        fidelity_id="hi_nerv",
        reachability_id="hi_nerv",
        constants_id="hi_nerv",
        owner="hinerv_completion (task #40)",
        pass_route="vendor-faithful PR95-HNeRV port OR patch (decided by #45 + atlas omega)",
        fail_route="skip recon-inert under MSE; needs grid-PE/skip ON + reachable objective",
        verdict_glob=("*hinerv*/hi_nerv_receiver_closed_modelsize_ladder.json",),
    ),
    VehicleSpec(
        vehicle="pact_nerv_vq",
        family="VQ-VAE NeRV (van den Oord 1711.00937 + HNeRV decoder)",
        fidelity_id="pact_nerv_vq",
        reachability_id="pact_nerv_vq",
        constants_id=None,
        owner="pact_nerv_vq_completion (task #44)",
        pass_route="skip + omega(measured) under scorer objective (NOT MSE); audit MLX-route VJP",
        fail_route="skip-free decoder mean-fields regardless of VQ; architecture under scorer-objective untested",
    ),
    VehicleSpec(
        vehicle="sane_hnerv",
        family="HNeRV-LC-v2 (PR101/PR100 canonical HNeRV with bilinear-skip)",
        fidelity_id="sane_hnerv",
        reachability_id=None,
        constants_id=None,
        owner="laundering remediation",
        pass_route="implement the advertised bilinear-skip OR correct the docstring",
        fail_route="documentation-fake: docstring claims bilinear-skip the forward never implements",
    ),
    VehicleSpec(
        vehicle="ff_nerv",
        family="FFNeRV-style band-limited DCT-grid NeRV",
        fidelity_id="ff_nerv",
        reachability_id=None,
        constants_id=None,
        owner="(dormant sketch)",
        pass_route="add a genuine HF residual path; 64x64 DCT grid cannot represent boundary HF",
        fail_route="skip-free + band-limited by construction -> strictly-worse mean-field variant",
    ),
    VehicleSpec(
        vehicle="pr110pp",
        family="PR110++ (selector/menu on a strong base carrier)",
        fidelity_id=None,
        reachability_id=None,
        constants_id=None,
        owner="pr110pp_r1_paired_eval / pr110pp_r2_nonmps_mode_table",
        pass_route="R1 paired contest-CPU eval confirms dS<=0 -> exact CandidateActionEvaluation row (L4)",
        fail_route="macOS-CPU vs Linux-x86_64-CPU per-mode ordering unstable -> no exact gain",
    ),
    VehicleSpec(
        vehicle="atlas_atoms_v3",
        family="Spectral atlas + V3 compiler (infrastructure)",
        fidelity_id=None,
        reachability_id=None,
        constants_id=None,
        owner="atlas_engine_mlx_jacobian / frozen_evaluator_contract",
        pass_route="measured law geometry + DeltaS-judge feed every vehicle's search/accept",
        fail_route="n/a — infrastructure, not a candidate program generator",
        is_vehicle=False,
    ),
)


@dataclass(frozen=True)
class VehicleRow:
    """One rendered dashboard row (per OS-doc "Dashboard discipline")."""

    vehicle: str
    family: str
    maturity_level: str
    maturity_name: str
    allowed_claim: str
    maturity_evidence: str
    latest_artifact: str
    latest_artifact_sha256: str
    authority_tier: str
    metric_family: str
    current_blocker: str
    next_command: str
    owner: str
    pass_route: str
    fail_route: str
    is_vehicle: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "vehicle": self.vehicle,
            "family": self.family,
            "maturity_level": self.maturity_level,
            "maturity_name": self.maturity_name,
            "allowed_claim": self.allowed_claim,
            "maturity_evidence": self.maturity_evidence,
            "latest_artifact": self.latest_artifact,
            "latest_artifact_sha256": self.latest_artifact_sha256,
            "authority_tier": self.authority_tier,
            "metric_family": self.metric_family,
            "current_blocker": self.current_blocker,
            "next_command": self.next_command,
            "owner": self.owner,
            "pass_route": self.pass_route,
            "fail_route": self.fail_route,
            "is_vehicle": self.is_vehicle,
        }


@dataclass(frozen=True)
class DashboardModel:
    """The full dashboard model (rows + live-work + frontier pointer)."""

    generated_at_utc: str
    repo_root: str
    rows: tuple[VehicleRow, ...]
    live_work: tuple[Mapping[str, object], ...]
    frontier: Mapping[str, object]
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_gaps: tuple[Mapping[str, object], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "pact_compiler_dashboard.v1",
            "generated_at_utc": self.generated_at_utc,
            "from_repo_root": self.repo_root,
            "rows": [r.as_dict() for r in self.rows],
            "live_work": [dict(w) for w in self.live_work],
            "frontier": dict(self.frontier),
            "notes": list(self.notes),
            "schema_gaps": [dict(g) for g in self.schema_gaps],
        }


# ---------------------------------------------------------------------------
# repo / SSD tier resolution
# ---------------------------------------------------------------------------


def default_repo_root() -> Path:
    """Repo root inferred from this file (src/comma_lab/...; 3 up)."""
    return Path(__file__).resolve().parents[2]


def ssd_tier_roots() -> tuple[Path, ...]:
    """Candidate SSD verdict roots in the CLAUDE.md storage-waterfall order."""
    return (
        Path("/Volumes/VertigoDataTier/pact"),
        Path("/Volumes/APDataStore/pact"),
    )


def _first_existing_ssd_glob(globs: Sequence[str]) -> tuple[str, str]:
    """Return ``(path, sha256)`` of the newest verdict matching any glob.

    Reads the verdict JSON's ``packet_sha256`` / ``candidate_archive_sha256`` /
    ``g1b_packet_sha256`` when present; otherwise the file's own content sha.
    Returns ``(AUDIT_PENDING, AUDIT_PENDING)`` when no SSD tier is mounted or
    no match exists (fail-soft: the SSD tier may be detached).
    """
    if not globs:
        return (AUDIT_PENDING, AUDIT_PENDING)
    best: tuple[float, Path] | None = None
    for root in ssd_tier_roots():
        if not root.is_dir():
            continue
        for pattern in globs:
            for match in root.glob(pattern):
                if not match.is_file():
                    continue
                mtime = match.stat().st_mtime
                if best is None or mtime > best[0]:
                    best = (mtime, match)
    if best is None:
        return (AUDIT_PENDING, AUDIT_PENDING)
    path = best[1]
    sha = AUDIT_PENDING
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in (
            "candidate_archive_sha256",
            "packet_sha256",
            "g1b_packet_sha256",
        ):
            val = payload.get(key) if isinstance(payload, Mapping) else None
            if isinstance(val, str) and val:
                sha = val
                break
    except (OSError, ValueError):
        pass
    return (str(path), sha)


# ---------------------------------------------------------------------------
# manifest loading (fail-soft)
# ---------------------------------------------------------------------------


def _load_fidelity(vehicle_id: str | None, repo_root: Path) -> VehicleFidelityManifest | None:
    if not vehicle_id:
        return None
    path = manifest_path_for_vehicle(vehicle_id, repo_root)
    if not path.is_file():
        return None
    try:
        return VehicleFidelityManifest.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, KeyError):
        return None


def _load_reachability(
    vehicle_id: str | None, repo_root: Path
) -> ObjectiveReachabilityManifest | None:
    if not vehicle_id:
        return None
    path = objective_reachability_path_for_vehicle(vehicle_id, repo_root)
    if not path.is_file():
        return None
    try:
        return ObjectiveReachabilityManifest.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, KeyError):
        return None


def _load_constants_declared_level(
    vehicle_id: str | None, repo_root: Path
) -> str | None:
    if not vehicle_id:
        return None
    path = (
        repo_root
        / ".omx"
        / "state"
        / "constants_provenance"
        / f"{vehicle_id}.json"
    )
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    lvl = payload.get("declared_maturity_level")
    return str(lvl) if isinstance(lvl, str) else None


# ---------------------------------------------------------------------------
# maturity assignment FROM EVIDENCE
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MaturityVerdict:
    level: str
    evidence: str
    authority_tier: str
    metric_family: str
    blocker: str
    latest_artifact: str = ""
    """Optional verdict-derived latest artifact path (overrides the SSD glob)."""


def _pr110pp_maturity(repo_root: Path) -> _MaturityVerdict:
    """PR110++ maturity from the byte-closure + no-op + advisory-delta proofs.

    L3 (archive-real) when a byte-closed candidate exists with a no-op
    consumption proof; L4 only when an exact CandidateActionEvaluation row with
    real d_seg/d_pose on a contest axis exists (the R1 paired eval). The R1 eval
    is dispatched (Modal call_id) but not yet a confirmed exact row -> L3 with
    "L4 in flight".
    """
    results = repo_root / "experiments" / "results"
    candidate_dirs = sorted(results.glob("pr110pp_r2_nonmps_candidate_*")) + sorted(
        results.glob("pr110pp_r1plus_candidate_*")
    )
    noop_proven = False
    byte_closed = False
    cand_path = ""
    for d in candidate_dirs:
        noop = d / "noop_detector.json"
        bcp = d / "byte_closure_proof.json"
        if noop.is_file():
            try:
                if json.loads(noop.read_text(encoding="utf-8")).get(
                    "consumption_proven"
                ):
                    noop_proven = True
            except (OSError, ValueError):
                pass
        if bcp.is_file():
            byte_closed = True
            cand_path = str(d / "candidate_archive.zip")
    r1_dirs = sorted(results.glob("pr110pp_r1_paired_eval_*"))
    r1_in_flight = any(
        (d / "strict_candidate_cpu" / "modal_call_id.txt").is_file() for d in r1_dirs
    )
    if byte_closed and noop_proven:
        blocker = (
            "L4 exact row in flight (R1 paired contest-CPU Modal eval dispatched); "
            "advisory pose gain only"
            if r1_in_flight
            else "no exact CandidateActionEvaluation row yet (run R1 paired eval)"
        )
        return _MaturityVerdict(
            level="L3",
            evidence=(
                f"byte_closure_proof.json + noop_detector.json(consumption_proven=true) "
                f"at {cand_path or 'pr110pp_r2_nonmps_candidate'}"
            ),
            authority_tier="exact_cpu_advisory",
            metric_family="advisory_pose_delta",
            blocker=blocker,
            latest_artifact=cand_path,
        )
    return _MaturityVerdict(
        level="L1",
        evidence="action_effect rows exist but no byte-closed candidate located",
        authority_tier=AUDIT_PENDING,
        metric_family=AUDIT_PENDING,
        blocker="byte-closed candidate + no-op proof not found on this checkout",
    )


def _snerv_maturity(
    fidelity: VehicleFidelityManifest | None,
    reach: ObjectiveReachabilityManifest | None,
    repo_root: Path,
) -> _MaturityVerdict:
    """SNeRV maturity from the G1b CandidateActionEvaluation row.

    L4 (exact-scored, advisory) when a CandidateActionEvaluation with real
    d_seg/d_pose + exact_pair_scorer metric exists — even at advisory authority,
    and even though it does not pay rent (the 581.6MB rate blocker). NOT L5+
    (no contest-axis paired row; pays_rent=false). Falls back to L1/L0 if the
    fidelity/reachability manifests are missing.
    """
    cae_authority = AUDIT_PENDING
    cae_metric = AUDIT_PENDING
    cae_evidence = ""
    for root in ssd_tier_roots():
        if not root.is_dir():
            continue
        for match in sorted(
            root.glob("snerv_mistake_b_g1a_*/candidate_action_evaluation_g1b_*.v1.json")
        ):
            try:
                payload = json.loads(match.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if payload.get("has_d_seg") and payload.get("has_d_pose"):
                cae_authority = str(payload.get("authority_tier", AUDIT_PENDING))
                cae_metric = str(payload.get("metric_family", AUDIT_PENDING))
                cae_evidence = (
                    f"{match.name}: d_seg={payload.get('candidate_d_seg')} "
                    f"d_pose={payload.get('candidate_d_pose')} "
                    f"bytes={payload.get('candidate_bytes')} "
                    f"pays_rent={payload.get('pays_rent')}"
                )
    fidelity_clean = bool(fidelity) and not (fidelity and fidelity.laundering_findings())
    reach_clean = bool(reach) and not (reach and reach.reachability_findings())
    if cae_evidence and fidelity_clean and reach_clean:
        return _MaturityVerdict(
            level="L4",
            evidence=(
                f"exact CandidateActionEvaluation: {cae_evidence}; "
                f"fidelity clean (MFU/HFR/TUB+DWT), reachability clean (VJPs reach)"
            ),
            authority_tier=cae_authority,
            metric_family=cae_metric,
            blocker=(
                "100% rate (581.6MB skip_high float64 LL=99.9996%); export bound but "
                "rate chasm -> route to LF entropy-coding front; NOT L5 (no contest-axis "
                "paired row, pays_rent=false)"
            ),
        )
    # No exact row located -> mechanism-present at best.
    if fidelity_clean and reach_clean:
        return _MaturityVerdict(
            level="L2",
            evidence="fidelity + reachability clean; no exact row located on this checkout",
            authority_tier=AUDIT_PENDING,
            metric_family=AUDIT_PENDING,
            blocker="G1b CandidateActionEvaluation not found (SSD tier detached?)",
        )
    return _MaturityVerdict(
        level="L1",
        evidence="fidelity/reachability manifests incomplete on this checkout",
        authority_tier=AUDIT_PENDING,
        metric_family=AUDIT_PENDING,
        blocker="manifests missing or failing",
    )


def _manifest_driven_maturity(
    spec: VehicleSpec,
    fidelity: VehicleFidelityManifest | None,
    reach: ObjectiveReachabilityManifest | None,
    constants_level: str | None,
) -> _MaturityVerdict:
    """Generic L0/L1 assignment from fidelity + reachability + constants level.

    * Laundering finding (documentation-fake) -> L0 (the name is not a claim).
    * No present mechanisms -> L0 (sketch).
    * >=1 genuine present mechanism but reachability fails (or no reachability
      manifest) -> L1 (mechanism-present; not intrinsically optimized).
    * The constants-provenance ``declared_maturity_level`` is surfaced when it
      DISAGREES with the evidence-derived level (operator-facing note).
    """
    if fidelity is None:
        return _MaturityVerdict(
            level="L0",
            evidence="no vehicle_fidelity manifest emitted (manifest_missing)",
            authority_tier=AUDIT_PENDING,
            metric_family=AUDIT_PENDING,
            blocker="emit a vehicle_fidelity_manifest.v1 to prove the carrier identity",
        )
    laundering = fidelity.laundering_findings()
    if laundering:
        return _MaturityVerdict(
            level="L0",
            evidence=f"NAME-LAUNDERING (documentation-fake): {laundering[0]}",
            authority_tier=AUDIT_PENDING,
            metric_family=AUDIT_PENDING,
            blocker="docstring claims a mechanism the forward never implements",
        )
    present = sorted(fidelity.present_mechanism_names())
    if not present:
        return _MaturityVerdict(
            level="L0",
            evidence="vehicle_fidelity manifest: zero present mechanisms (honest sketch)",
            authority_tier=AUDIT_PENDING,
            metric_family=AUDIT_PENDING,
            blocker="no HF residual path / objective mechanism present",
        )
    # >=1 present mechanism -> L1 (mechanism-present). Reachability state shapes
    # the blocker.
    reach_blocker = "objective reachability manifest not emitted"
    if reach is not None:
        findings = reach.reachability_findings()
        if findings:
            reach_blocker = f"objective severance at {reach.first_failed_surface or 'unknown'} surface"
        else:
            reach_blocker = "reachability clean; intrinsic L2 native task not yet proven"
    evidence = (
        f"vehicle_fidelity present mechanisms {present} (file:line evidence); "
        f"L1 mechanism-present"
    )
    if constants_level and constants_level != "L1":
        evidence += f" [constants_provenance declares {constants_level}]"
    return _MaturityVerdict(
        level="L1",
        evidence=evidence,
        authority_tier=AUDIT_PENDING,
        metric_family=AUDIT_PENDING,
        blocker=reach_blocker,
    )


def _next_command(vehicle: str) -> str:
    """The OS-doc next decisive command per vehicle."""
    return {
        "snerv": (
            ".venv/bin/python tools/... (LF entropy-coding rate attack; "
            "branch B ladder on snerv_branch_b_rate_attack)"
        ),
        "hi_nerv": (
            ".venv/bin/python -m pytest src/tac/substrates/hi_nerv/ "
            "(then vendor PR95-HNeRV port vs patch decision per #40)"
        ),
        "pact_nerv_vq": (
            "audit MLX-route scorer-objective VJP; set nonzero SegNet/PoseNet "
            "weights; skip+omega under scorer objective (NOT MSE)"
        ),
        "sane_hnerv": (
            "implement advertised bilinear-skip in forward OR correct the docstring "
            "(architecture.py:5,27,122)"
        ),
        "ff_nerv": "(dormant) add a genuine HF residual path before any training run",
        "pr110pp": (
            ".venv/bin/python tools/... harvest R1 paired contest-CPU Modal eval "
            "(strict_candidate_cpu/modal_call_id.txt) -> exact CandidateActionEvaluation"
        ),
        "atlas_atoms_v3": (
            "atlas v2 full sweep (running) -> learned-omega Nyquist-capped basis; "
            "V3 ingest of #45's row"
        ),
    }.get(vehicle, AUDIT_PENDING)


# ---------------------------------------------------------------------------
# row + model construction
# ---------------------------------------------------------------------------


def _build_row(spec: VehicleSpec, repo_root: Path) -> VehicleRow:
    if not spec.is_vehicle:
        # Infrastructure row — n/a-vehicle.
        latest, sha = _first_existing_ssd_glob(spec.verdict_glob)
        return VehicleRow(
            vehicle=spec.vehicle,
            family=spec.family,
            maturity_level="n/a-vehicle",
            maturity_name="infrastructure",
            allowed_claim="infrastructure (not a candidate program generator)",
            maturity_evidence=(
                "spectral atlas (evaluator_response_atlas.py) + V3 "
                "(frozen_evaluator_contract.py) are the measured-law + DeltaS-judge "
                "kernels every vehicle consumes; not on the L0-L7 ladder"
            ),
            latest_artifact=latest,
            latest_artifact_sha256=sha,
            authority_tier="n/a",
            metric_family="n/a",
            current_blocker="n/a — infrastructure",
            next_command=_next_command(spec.vehicle),
            owner=spec.owner,
            pass_route=spec.pass_route,
            fail_route=spec.fail_route,
            is_vehicle=False,
        )

    fidelity = _load_fidelity(spec.fidelity_id, repo_root)
    reach = _load_reachability(spec.reachability_id, repo_root)
    constants_level = _load_constants_declared_level(spec.constants_id, repo_root)

    if spec.vehicle == "pr110pp":
        verdict = _pr110pp_maturity(repo_root)
    elif spec.vehicle == "snerv":
        verdict = _snerv_maturity(fidelity, reach, repo_root)
    else:
        verdict = _manifest_driven_maturity(spec, fidelity, reach, constants_level)

    latest, sha = _first_existing_ssd_glob(spec.verdict_glob)
    if verdict.latest_artifact:
        # A verdict-derived artifact (e.g. pr110pp candidate dir) takes precedence.
        latest = verdict.latest_artifact
    elif latest == AUDIT_PENDING and fidelity is not None:
        # Fall back to the fidelity manifest as the latest durable artifact.
        latest = str(manifest_path_for_vehicle(spec.fidelity_id, repo_root))

    allowed = _MATURITY_CLAIM.get(verdict.level, verdict.level)
    return VehicleRow(
        vehicle=spec.vehicle,
        family=spec.family,
        maturity_level=verdict.level,
        maturity_name=_MATURITY_NAME.get(verdict.level, verdict.level),
        allowed_claim=allowed,
        maturity_evidence=verdict.evidence,
        latest_artifact=latest,
        latest_artifact_sha256=sha,
        authority_tier=verdict.authority_tier,
        metric_family=verdict.metric_family,
        current_blocker=verdict.blocker,
        next_command=_next_command(spec.vehicle),
        owner=spec.owner,
        pass_route=spec.pass_route,
        fail_route=spec.fail_route,
        is_vehicle=True,
    )


def _load_live_work(repo_root: Path, *, limit: int = 12) -> tuple[Mapping[str, object], ...]:
    """Latest in-flight / running subagents from subagent_progress.jsonl.

    Returns the most-recent row PER subagent_id whose latest status is
    ``in_progress`` or ``blocked`` (running daemons/agents), newest first.
    Fail-soft: a missing/corrupt log yields an empty tuple.
    """
    path = repo_root / ".omx" / "state" / "subagent_progress.jsonl"
    if not path.is_file():
        return ()
    latest_by_id: dict[str, Mapping[str, object]] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            sid = str(row.get("subagent_id", ""))
            if not sid:
                continue
            latest_by_id[sid] = row  # latest-row-wins (file is append-only)
    except OSError:
        return ()
    running = [
        {
            "subagent_id": r.get("subagent_id", ""),
            "status": r.get("status", ""),
            "step": r.get("step", ""),
            "next_action": r.get("next_action", ""),
            "written_at_utc": r.get("written_at_utc", ""),
        }
        for r in latest_by_id.values()
        if str(r.get("status", "")).lower() in {"in_progress", "blocked"}
    ]
    running.sort(key=lambda r: str(r.get("written_at_utc", "")), reverse=True)
    return tuple(running[:limit])


def _load_frontier(repo_root: Path) -> Mapping[str, object]:
    """Pointer-only frontier (NEVER hardcode scores per CLAUDE.md)."""
    path = repo_root / ".omx" / "state" / "canonical_frontier_pointer.json"
    if not path.is_file():
        return {
            "status": "POINTER_MISSING",
            "note": "canonical_frontier_pointer.json not found; scores are pointer-only",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"status": "POINTER_UNREADABLE", "error": str(exc)}
    cpu = payload.get("our_local_frontier_contest_cpu", {}) or {}
    cuda = payload.get("our_local_frontier_contest_cuda", {}) or {}
    return {
        "status": "ok",
        "contest_cpu_score": cpu.get("score", AUDIT_PENDING),
        "contest_cpu_archive_sha256": cpu.get("archive_sha256", AUDIT_PENDING),
        "contest_cuda_score": cuda.get("score", AUDIT_PENDING),
        "contest_cuda_archive_sha256": cuda.get("archive_sha256", AUDIT_PENDING),
        "submitted_pr_number_for_current_frontier": payload.get(
            "submitted_pr_number_for_current_frontier"
        ),
        "last_refreshed_utc": payload.get("last_refreshed_utc", AUDIT_PENDING),
        "pointer_path": str(path),
    }


def build_dashboard_model(
    repo_root: str | Path | None = None,
    *,
    generated_at_utc: str | None = None,
) -> DashboardModel:
    """Assemble the full dashboard model from the machine-readable sources."""
    root = Path(repo_root) if repo_root is not None else default_repo_root()
    if generated_at_utc is None:
        generated_at_utc = datetime.now(UTC).isoformat()
    rows = tuple(_build_row(spec, root) for spec in VEHICLE_SPECS)
    live = _load_live_work(root)
    frontier = _load_frontier(root)
    notes = (
        "Maturity is assigned FROM EVIDENCE (see each row's maturity_evidence); "
        "no level is asserted without a machine-readable basis.",
        "Scores are POINTER-ONLY (canonical_frontier_pointer.json); never hardcoded.",
        "SCHEMA GAP: vehicle_fidelity_manifest.v1 has no maturity_level field; the "
        "canonical per-vehicle declared maturity lives in constants_provenance "
        "(declared_maturity_level) where a constants manifest exists (only hi_nerv "
        "today). The dashboard derives maturity from the union of fidelity + "
        "reachability + constants + the typed verdict rows.",
        "snerv fidelity manifest id is 'snerv_inverse_steg_carrier'; reachability id "
        "is 'snerv' — bound via VehicleSpec.fidelity_id/reachability_id.",
    )
    schema_gaps = (
        {
            "surface": "vehicle_fidelity_manifest.v1",
            "gap": "no maturity_level field",
            "consequence": (
                "maturity cannot be recorded in-manifest via the canonical emitter; "
                "the dashboard derives it from fidelity + reachability + constants + "
                "typed verdict rows instead."
            ),
            "remediation": (
                "council/operator-approved schema addition (a maturity_level field on "
                "VehicleFidelityManifest) — a design decision, NOT a hand-edit; OR "
                "standardize on constants_provenance.declared_maturity_level for ALL "
                "vehicles (only hi_nerv has a constants manifest today)."
            ),
        },
    )
    return DashboardModel(
        generated_at_utc=generated_at_utc,
        repo_root=str(root),
        rows=rows,
        live_work=live,
        frontier=frontier,
        notes=notes,
        schema_gaps=schema_gaps,
    )


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render_json(model: DashboardModel) -> str:
    return json.dumps(model.as_dict(), indent=2, sort_keys=False) + "\n"


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(model: DashboardModel) -> str:
    lines: list[str] = []
    lines.append("# PACT compiler dashboard (Vehicle-OS)")
    lines.append("")
    lines.append(
        f"Generated {model.generated_at_utc} from `{model.repo_root}`. "
        "Per `docs/vehicle_operating_system.md` Dashboard discipline: work is "
        "dashboard-driven; no stale-memory decisions."
    )
    lines.append("")

    # Frontier (pointer-only).
    f = model.frontier
    lines.append("## Frontier (pointer-only — never hardcoded)")
    lines.append("")
    if f.get("status") == "ok":
        lines.append(
            f"- contest-CPU: **{f.get('contest_cpu_score')}** "
            f"(`{str(f.get('contest_cpu_archive_sha256'))[:12]}`)"
        )
        lines.append(
            f"- contest-CUDA: **{f.get('contest_cuda_score')}** "
            f"(`{str(f.get('contest_cuda_archive_sha256'))[:12]}`)"
        )
        lines.append(
            f"- submitted PR for current frontier: "
            f"`{f.get('submitted_pr_number_for_current_frontier')}`"
        )
        lines.append(f"- pointer last refreshed: {f.get('last_refreshed_utc')}")
    else:
        lines.append(f"- **{f.get('status')}**: {f.get('note', f.get('error', ''))}")
    lines.append("")

    # Per-vehicle table.
    lines.append("## Per-vehicle maturity (L0-L7)")
    lines.append("")
    header = (
        "| vehicle | L | allowed_claim | latest_artifact (sha) | authority_tier | "
        "metric_family | current_blocker | next_command | owner | pass_route | fail_route |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 11)
    for r in model.rows:
        sha_short = (
            r.latest_artifact_sha256[:12]
            if r.latest_artifact_sha256 not in {"", AUDIT_PENDING}
            else r.latest_artifact_sha256
        )
        artifact = _md_cell(r.latest_artifact)
        if sha_short:
            artifact = f"{artifact} ({sha_short})"
        lines.append(
            "| "
            + " | ".join(
                _md_cell(c)
                for c in (
                    r.vehicle,
                    r.maturity_level,
                    r.allowed_claim,
                    artifact,
                    r.authority_tier,
                    r.metric_family,
                    r.current_blocker,
                    r.next_command,
                    r.owner,
                    r.pass_route,
                    r.fail_route,
                )
            )
            + " |"
        )
    lines.append("")

    # Maturity evidence (one line per vehicle — the cite-per-assignment).
    lines.append("## Maturity evidence (FROM EVIDENCE — cite per assignment)")
    lines.append("")
    for r in model.rows:
        lines.append(f"- **{r.vehicle}** {r.maturity_level}: {r.maturity_evidence}")
    lines.append("")

    # Live work.
    lines.append("## Live work (running daemons/agents)")
    lines.append("")
    if model.live_work:
        lines.append("| subagent_id | status | step | next_action | written_at_utc |")
        lines.append("|---|---|---|---|---|")
        for w in model.live_work:
            lines.append(
                "| "
                + " | ".join(
                    _md_cell(w.get(k, ""))
                    for k in ("subagent_id", "status", "step", "next_action", "written_at_utc")
                )
                + " |"
            )
    else:
        lines.append("_No in-progress subagents in the progress log._")
    lines.append("")

    # Schema gaps (machine-readable; per the "note the schema gap" directive).
    if model.schema_gaps:
        lines.append("## Schema gaps")
        lines.append("")
        for g in model.schema_gaps:
            lines.append(
                f"- **{g.get('surface')}**: {g.get('gap')} — {g.get('consequence')} "
                f"_Remediation_: {g.get('remediation')}"
            )
        lines.append("")

    # Notes.
    lines.append("## Notes")
    lines.append("")
    for n in model.notes:
        lines.append(f"- {n}")
    lines.append("")
    return "\n".join(lines)


def write_dashboard(
    repo_root: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
    generated_at_utc: str | None = None,
) -> tuple[Path, Path]:
    """Build + write ``pact_compiler_dashboard.{json,md}``; return both paths."""
    root = Path(repo_root) if repo_root is not None else default_repo_root()
    out = Path(out_dir) if out_dir is not None else root
    model = build_dashboard_model(root, generated_at_utc=generated_at_utc)
    json_path = out / "pact_compiler_dashboard.json"
    md_path = out / "pact_compiler_dashboard.md"
    json_path.write_text(render_json(model), encoding="utf-8")
    md_path.write_text(render_markdown(model), encoding="utf-8")
    return (json_path, md_path)
