"""Static registry of partially built hidden-gem techniques.

This module is intentionally read-only and import-light. It makes unfinished
but promising local techniques discoverable without inspecting provider state,
claiming scores, or launching GPU work.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath

SCHEMA_VERSION = 1

CATEGORIES = frozenset(
    {
        "archive_packing",
        "entropy_coding",
        "geometry_pose",
        "latent_repair",
        "mask_representation",
        "renderer_payload",
        "stack_composition",
    }
)

STATUSES = frozenset(
    {
        "planning",
        "prototype",
        "ready_for_patch",
        "implemented_guarded",
    }
)

SENSITIVE_PATH_PREFIXES = (
    ".omx/state/",
    "experiments/results/",
    "reports/private/",
    "reports/raw/",
    "reverse_engineering/",
)

SENSITIVE_PATH_PARTS = frozenset(
    {
        ".env",
        "credential",
        "credentials",
        "id_rsa",
        "provider_state",
        "secret",
        "secrets",
        "ssh",
        "token",
        "tokens",
    }
)

CONCRETE_SCORE_CLAIM_PATTERNS = (
    re.compile(r"\bscore_recomputed_from_components\b", re.IGNORECASE),
    re.compile(r"\bfinal_score\b", re.IGNORECASE),
    re.compile(r"\bavg_posenet_dist\b", re.IGNORECASE),
    re.compile(r"\bavg_segnet_dist\b", re.IGNORECASE),
    re.compile(r"\bleaderboard\s+rank\b", re.IGNORECASE),
    re.compile(r"\brank\s*#?\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bscore\s*[=:]\s*[-+]?\d", re.IGNORECASE),
)

KEY_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_]*[a-z0-9])?$")


@dataclass(frozen=True)
class HiddenGemEntry:
    """One deterministic hidden-gem registry row."""

    key: str
    title: str
    category: str
    status: str
    summary: str
    evidence_paths: tuple[str, ...]
    integration_targets: tuple[str, ...]
    next_patch: str
    contest_compliance_notes: tuple[str, ...]


_REGISTRY: tuple[HiddenGemEntry, ...] = (
    HiddenGemEntry(
        key="charged_mask_grammar_atoms",
        title="Charged mask grammar atoms",
        category="mask_representation",
        status="planning",
        summary=(
            "Turn the charged mask grammar and ego-foveation notes into a byte-accounted "
            "local atom manifest before any renderer integration."
        ),
        evidence_paths=(
            ".omx/research/charged_mask_grammar_atoms_20260502_codex.md",
            ".omx/research/charged_mask_grammar_ego_foveation_greenup_20260502_codex.md",
            "src/tac/hyperbolic_foveation.py",
            "src/tac/tests/test_hyperbolic_foveation.py",
        ),
        integration_targets=(
            "src/tac/mask_entropy_coder.py",
            "src/tac/hyperbolic_foveation.py",
        ),
        next_patch=(
            "Define a charged atom manifest that separates syntax bytes, decoder bytes, "
            "and frozen input streams before renderer integration."
        ),
        contest_compliance_notes=(
            "Manifest must avoid hidden side channels and must record every charged byte source.",
            "Planning artifact only; exact archive custody is required before promotion.",
        ),
    ),
    HiddenGemEntry(
        key="component_sensitivity_byte_allocator",
        title="Component sensitivity byte allocator",
        category="renderer_payload",
        status="implemented_guarded",
        summary=(
            "Use certified component-sensitivity artifacts to plan renderer payload byte allocation "
            "without reading provider queues or launching eval work."
        ),
        evidence_paths=(
            ".omx/research/component_sensitivity_map_certification_20260501_codex.md",
            ".omx/research/component_sensitivity_byte_plan_manifest_20260506_codex.md",
            ".omx/research/owv3_fisher_byte_aware_redesign_spec_20260430_codex.md",
            "src/tac/component_sensitivity_artifact.py",
            "src/tac/owv3_sensitivity_weighted.py",
            "src/tac/tests/test_sweep_owv3_byte_plan.py",
        ),
        integration_targets=(
            "experiments/sweep_owv3_byte_plan.py",
            "src/tac/owv3_sensitivity_weighted.py",
        ),
        next_patch=(
            "Use `experiments/sweep_owv3_byte_plan.py --manifest-only` as the fast "
            "planning preflight, then run the full sweep only with authoritative CUDA "
            "sensitivity custody."
        ),
        contest_compliance_notes=(
            "Local byte analysis only; exact archive eval is required before promotion.",
            "Do not reference provider workspaces or state files from the manifest.",
        ),
    ),
    HiddenGemEntry(
        key="coolchic_c3_renderer_trend_gate",
        title="Cool-Chic C3 renderer trend gate",
        category="renderer_payload",
        status="prototype",
        summary=(
            "Promote Cool-Chic and C3 renderer prototypes from local wiring smoke to a "
            "controlled CUDA trend gate before archive integration."
        ),
        evidence_paths=(
            "reports/local_smoke_coolchic_c3_20260425.md",
            "src/tac/contrib/coolchic_renderer.py",
            "src/tac/contrib/coolchic_darts.py",
            "src/tac/profiles.py",
            "src/tac/experiments/train_renderer.py",
        ),
        integration_targets=(
            "tools/list_hidden_gems.py",
            "src/tac/experiments/train_renderer.py",
        ),
        next_patch=(
            "Add a deterministic CUDA smoke-matrix runner and block archive wiring until "
            "trend metadata proves non-regression."
        ),
        contest_compliance_notes=(
            "Smoke and trend artifacts are development evidence only.",
            "Renderer payload changes need byte-closed archives and exact CUDA replay.",
        ),
    ),
    HiddenGemEntry(
        key="engineered_correction_atom_gate",
        title="Engineered correction atom gate",
        category="latent_repair",
        status="ready_for_patch",
        summary=(
            "Make AMR-style and pixel-diff correction atoms no-op-resistant, signed by "
            "component traces, and fail-closed before dispatch."
        ),
        evidence_paths=(
            ".omx/research/pr85_correction_atom_waterfill_worker_20260504.md",
            "experiments/precompute_gradient_corrections.py",
            "src/tac/engineered_correction_readiness.py",
            "src/tac/engineered_corrections.py",
            "src/tac/engineered_corrections_v2.py",
        ),
        integration_targets=(
            "tools/audit_engineered_corrections.py",
            "tools/all_lanes_preflight.py",
            "src/tac/engineered_corrections_v2.py",
            "experiments/precompute_gradient_corrections.py",
        ),
        next_patch=(
            "Feed a real component-trace manifest into the guarded local patch path; exact "
            "dispatch remains blocked until a byte-closed archive consumes charged atoms."
        ),
        contest_compliance_notes=(
            "No correction atom can promote from proxy or byte-only evidence.",
            "Every runtime-applied atom must be charged inside the archive.",
        ),
    ),
    HiddenGemEntry(
        key="fridrich_inverse_steg_allocator",
        title="Fridrich inverse-steg allocator",
        category="latent_repair",
        status="ready_for_patch",
        summary=(
            "Use UNIWARD/Fridrich-style detector-aware costs as allocation feedback for "
            "charged residual atoms and HNeRV section transforms."
        ),
        evidence_paths=(
            ".omx/research/council_strategic_design_decisions_20260430.md",
            ".omx/research/council_uniward_v8_fridrich_shannon_audit_20260429.md",
            "src/tac/fridrich.py",
            "src/tac/fridrich_losses.py",
            "src/tac/uniward_texture.py",
            "src/tac/tests/test_uniward_texture.py",
        ),
        integration_targets=(
            "src/tac/hnerv_section_repack.py",
            "src/tac/engineered_correction_readiness.py",
            "src/tac/uniward_delta.py",
        ),
        next_patch=(
            "Emit a detector-cost field that ranks HNeRV latent, sidecar, and correction "
            "atoms by charged bytes and scorer-sensitivity evidence."
        ),
        contest_compliance_notes=(
            "Fridrich/UNIWARD maps are optimizer feedback only until consumed by charged archive bytes.",
            "No prior UNIWARD result promotes without proving the computed payload reached inflate.",
        ),
    ),
    HiddenGemEntry(
        key="hnerv_payload_scorecard_followups",
        title="HNeRV payload scorecard follow-ups",
        category="archive_packing",
        status="ready_for_patch",
        summary=(
            "Make public HNeRV payload anatomy drive deterministic repack follow-ups from "
            "section bytes and provenance instead of prose."
        ),
        evidence_paths=(
            ".omx/research/public_hnerv_frontier_deconstruction_20260504_codex.md",
            "experiments/build_hnerv_frontier_scorecard.py",
            "experiments/profile_hnerv_frontier_payloads.py",
        ),
        integration_targets=(
            "src/tac/archive_byte_profile.py",
            "src/tac/public_submission_refs.py",
        ),
        next_patch=(
            "Use payload-section manifests and source/candidate HNeRV section diffs as "
            "mandatory no-op controls for every future HNeRV repack candidate."
        ),
        contest_compliance_notes=(
            "Byte forensics only; no archive is eligible for dispatch until exact replay gates pass.",
            "Candidate manifests must preserve archive and member hashes.",
        ),
    ),
    HiddenGemEntry(
        key="joint_stack_contract_manifest",
        title="Joint stack contract manifest",
        category="stack_composition",
        status="planning",
        summary=(
            "Record representation, prediction, quantization, hyperprior, arithmetic, and pack "
            "stages as typed local contracts before stacking experiments."
        ),
        evidence_paths=(
            ".omx/research/contest_faithful_swarm_execution_20260502_codex.md",
            ".omx/research/works_negatives_hardened_stack_20260502_codex.md",
            "src/tac/joint_admm_coordinator.py",
            "src/tac/stack_compositions.py",
        ),
        integration_targets=(
            "src/tac/joint_codec_stack_orchestrator.py",
            "src/tac/stack_compositions.py",
        ),
        next_patch=(
            "Add a typed local manifest that records each stack stage and its byte contract "
            "without dispatch metadata."
        ),
        contest_compliance_notes=(
            "No provider state, provider logs, or live job identifiers belong in this manifest.",
            "Stack deltas remain planning evidence until exact archive custody exists.",
        ),
    ),
    HiddenGemEntry(
        key="latent_sidecar_arithmetic_terminal",
        title="Latent sidecar arithmetic terminal",
        category="entropy_coding",
        status="prototype",
        summary=(
            "Reuse static arithmetic-coding helpers for latent-correction sidecar bytes after "
            "proving byte-for-byte decode parity on fixtures."
        ),
        evidence_paths=(
            ".omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md",
            "src/tac/arithmetic_qint_codec.py",
            "src/tac/entropy_archive.py",
            "src/tac/tests/test_arithmetic_qint_codec.py",
        ),
        integration_targets=(
            "src/tac/archive_byte_profile.py",
            "src/tac/entropy_archive.py",
        ),
        next_patch=(
            "Port static-model arithmetic pack and unpack helpers onto HNeRV latent-correction "
            "sidecar bytes with a parity-first fixture."
        ),
        contest_compliance_notes=(
            "Side information must be charged inside the packed payload.",
            "Inflate must remain scorer-free and deterministic.",
        ),
    ),
    HiddenGemEntry(
        key="nerv_mask_l2_readiness",
        title="NeRV mask L2 readiness",
        category="mask_representation",
        status="planning",
        summary=(
            "Recover the NeRV mask codec lane as a guarded build-only path until L2 "
            "clearance, parser closure, and pose provenance are present."
        ),
        evidence_paths=(
            ".omx/research/lane12_nerv_l2_unblock_worker_e_20260430.md",
            "src/tac/nerv_mask_codec.py",
            "experiments/train_nerv_mask.py",
            "scripts/remote_lane_nerv.sh",
        ),
        integration_targets=(
            "tools/list_hidden_gems.py",
            "src/tac/nerv_mask_codec.py",
        ),
        next_patch=(
            "Add a dry-run readiness command that checks L2 clearance, runtime dispatch, "
            "dependency closure, and exact archive-builder availability."
        ),
        contest_compliance_notes=(
            "No training run or archive can promote without the L2 clearance artifact.",
            "Learned mask bytes and decoder state must be charged inside the archive.",
        ),
    ),
    HiddenGemEntry(
        key="omega_w_v3_real_sensitivity_gate",
        title="Omega-W-V3 real sensitivity gate",
        category="stack_composition",
        status="implemented_guarded",
        summary=(
            "Make the PR106 water-fill repack lane refuse stub, stale, or mismatched "
            "sensitivity metadata before production dispatch."
        ),
        evidence_paths=(
            ".omx/research/contest_faithful_swarm_execution_20260502_codex.md",
            "experiments/repack_pr106_with_water_filling.py",
            "src/tac/water_filling_codec_v2.py",
            "tools/dispatch_dryrun_omega_w_v3.py",
        ),
        integration_targets=(
            "tools/dispatch_dryrun_omega_w_v3.py",
            "tools/all_lanes_preflight.py",
        ),
        next_patch=(
            "Wire certified sensitivity maps into a concrete OWV3 candidate builder "
            "and keep exact CUDA eval as the only promotion path."
        ),
        contest_compliance_notes=(
            "Sensitivity is optimizer feedback, not score evidence.",
            "Production dispatch needs exact source archive identity, CUDA device "
            "provenance, and component_sensitivity_map_certification_v1 metadata.",
        ),
    ),
    HiddenGemEntry(
        key="pr106_sidechannel_stack_gate",
        title="PR106 sidechannel stack gate",
        category="stack_composition",
        status="implemented_guarded",
        summary=(
            "Coordinate latent, yshift, LRL1, and stacked PR106 sidechannels through one "
            "dry-run gate so sister archives cannot drift or dispatch from proxy-only modes."
        ),
        evidence_paths=(
            "experiments/build_pr106_latent_sidecar.py",
            "experiments/build_pr106_yshift_sidechannel.py",
            "experiments/build_pr106_lrl1_sidechannel.py",
            "experiments/build_pr106_stacked.py",
            "tools/operator_briefing.py",
        ),
        integration_targets=(
            "tools/all_lanes_preflight.py",
            "tools/operator_briefing.py",
        ),
        next_patch=(
            "Implement one CUDA scorer-backed yshift or latent search mode, then feed its "
            "exact CUDA artifact back into the stack gate."
        ),
        contest_compliance_notes=(
            "Proxy self-consistency cannot unlock exact-eval dispatch by itself.",
            "Production readiness proves sister archives are anchored to the same PR106 "
            "payload, manifests match archive paths and bytes, and stacked dispatch "
            "blockers remain explicit until exact CUDA auth eval.",
        ),
    ),
    HiddenGemEntry(
        key="pr91_hpm1_runtime_contract",
        title="PR91 HPM1 runtime contract",
        category="mask_representation",
        status="ready_for_patch",
        summary=(
            "Keep the public PR91 HPM1 mask stream visible as a high-EV categorical "
            "rate signal while its HPAC device contract, full decode, and re-encode "
            "parity remain fail-closed."
        ),
        evidence_paths=(
            ".omx/research/pr91_hpm1_readiness_20260506_codex.md",
            "src/tac/pr91_hpm1_codec.py",
            "src/tac/pr91_hpm1_readiness.py",
            "src/tac/pr91_hpm1_runtime_contract.py",
            "src/tac/tests/test_pr91_hpm1_runtime_contract.py",
        ),
        integration_targets=(
            "tools/audit_pr91_hpm1_readiness.py",
            "tools/audit_pr91_hpm1_runtime_contract.py",
            "experiments/replay_pr91_hpm1_mask.py",
            "experiments/preflight_pr91_pr92_replay_contracts.py",
        ),
        next_patch=(
            "Resolve the HPAC CPU/CUDA device contract, then recover full HPM1 "
            "decode/reencode parity before any archive mutation or lane dispatch."
        ),
        contest_compliance_notes=(
            "Static custody and runtime-contract audits are not score evidence.",
            "No PR91/HPM1 candidate may dispatch until the HPM1 runtime consumes charged bytes "
            "without uncharged sidecars or fallback.",
        ),
    ),
    HiddenGemEntry(
        key="pr95_residual_atom_planner",
        title="PR95 residual atom planner",
        category="latent_repair",
        status="prototype",
        summary=(
            "Turn PR95-family residual atom notes into a tracked no-op-resistant candidate-member "
            "diff workflow."
        ),
        evidence_paths=(
            ".omx/research/public_hnerv_adapter_replays_20260504_codex.md",
            ".omx/research/public_hnerv_frontier_deconstruction_20260504_codex.md",
            "experiments/build_hnerv_frontier_scorecard.py",
            "experiments/profile_hnerv_frontier_payloads.py",
        ),
        integration_targets=(
            "src/tac/archive_byte_profile.py",
            "src/tac/public_submission_refs.py",
        ),
        next_patch=(
            "Create a tracked PR95-family atom planner with a no-op-resistant fixture and "
            "candidate-member diff report before any archive rebuild."
        ),
        contest_compliance_notes=(
            "Sidecars must stay forbidden and every atom plan must preserve source hashes.",
            "Candidate archives need exact replay custody before promotion.",
        ),
    ),
    HiddenGemEntry(
        key="raft_radial_pose_readiness",
        title="RAFT radial pose readiness",
        category="geometry_pose",
        status="prototype",
        summary=(
            "Use RAFT and radial ego-motion as pose-sidecar or TTO-initializer feedback only "
            "after disagreement and runtime-consumption proofs exist."
        ),
        evidence_paths=(
            "src/tac/raft_pose.py",
            "src/tac/raft_radial_pose.py",
            "experiments/compute_raft_flow.py",
            "experiments/derive_poses_from_raft.py",
            "scripts/remote_lane_fl_raft_derived_poses.sh",
        ),
        integration_targets=(
            "src/tac/raft_radial_pose.py",
            "experiments/derive_poses_from_raft.py",
        ),
        next_patch=(
            "Add a RAFT readiness command that emits a pose-disagreement artifact and "
            "refuses inflate/runtime modes without that proof."
        ),
        contest_compliance_notes=(
            "RAFT outputs are profile feedback until charged bytes consume them in inflate.",
            "Small geometry errors require exact CUDA component gates before promotion.",
        ),
    ),
    HiddenGemEntry(
        key="wavelet_residual_basis_gate",
        title="Wavelet residual basis gate",
        category="latent_repair",
        status="prototype",
        summary=(
            "Use wavelet bases as compact residual or mask-side correction coordinates, "
            "not as an unchecked replacement for the current HNeRV anchor."
        ),
        evidence_paths=(
            ".omx/research/contest_grade_all_lane_results_audit_20260430.md",
            ".omx/research/council_strategic_design_decisions_20260430.md",
            "src/tac/wavelet_mask_codec.py",
            "src/tac/wavelet_variance.py",
            "src/tac/contrib/wavelet_renderer.py",
            "src/tac/hnerv_wavelet_residual.py",
            "src/tac/hnerv_wavelet_sidechannel.py",
            "src/tac/hnerv_wavelet_apply_gate.py",
            "src/tac/hnerv_wavelet_apply_transform.py",
            "src/tac/tests/test_wavelet_mask_codec.py",
            "src/tac/tests/test_hnerv_wavelet_residual.py",
            "src/tac/tests/test_hnerv_wavelet_sidechannel.py",
            "src/tac/tests/test_hnerv_wavelet_apply_gate.py",
            "src/tac/tests/test_hnerv_wavelet_apply_transform.py",
            ".omx/research/hnerv_wavelet_sidechannel_candidate_20260506_codex.md",
        ),
        integration_targets=(
            "src/tac/hnerv_section_repack.py",
            "src/tac/archive_byte_profile.py",
            "src/tac/wavelet_mask_codec.py",
            "tools/plan_hnerv_wavelet_residual.py",
            "tools/build_hnerv_wavelet_sidechannel_candidate.py",
            "tools/audit_hnerv_wavelet_apply_gate.py",
            "tools/build_hnerv_wavelet_apply_transform_candidate.py",
        ),
        next_patch=(
            "Claim lane dispatch, run exact CUDA auth eval on the byte-negative "
            "offline WR01 half-strength candidate, then harvest and adjudicate before promotion."
        ),
        contest_compliance_notes=(
            "Wavelet coefficients are charged side information unless generated by fixed contest code.",
            "Exact CUDA archive eval is required before ranking any wavelet residual transform.",
        ),
    ),
)


def hidden_gem_to_dict(entry: HiddenGemEntry) -> dict[str, object]:
    """Return a JSON-stable mapping for one registry row."""
    return {
        "category": entry.category,
        "contest_compliance_notes": list(entry.contest_compliance_notes),
        "evidence_paths": list(entry.evidence_paths),
        "integration_targets": list(entry.integration_targets),
        "key": entry.key,
        "next_patch": entry.next_patch,
        "status": entry.status,
        "summary": entry.summary,
        "title": entry.title,
    }


def all_hidden_gems(
    *,
    category: str | None = None,
    status: str | None = None,
) -> tuple[HiddenGemEntry, ...]:
    """Return registry rows in deterministic key order, optionally filtered."""
    if category is not None and category not in CATEGORIES:
        raise ValueError(f"unknown hidden-gem category: {category!r}")
    if status is not None and status not in STATUSES:
        raise ValueError(f"unknown hidden-gem status: {status!r}")
    entries = _REGISTRY
    if category is not None:
        entries = tuple(entry for entry in entries if entry.category == category)
    if status is not None:
        entries = tuple(entry for entry in entries if entry.status == status)
    return entries


def registry_payload(entries: Iterable[HiddenGemEntry] | None = None) -> dict[str, object]:
    """Return the deterministic JSON payload for registry consumers."""
    rows = tuple(all_hidden_gems() if entries is None else entries)
    return {
        "entries": [hidden_gem_to_dict(entry) for entry in rows],
        "entry_count": len(rows),
        "registry": "hidden_gems",
        "schema_version": SCHEMA_VERSION,
    }


def render_markdown(entries: Iterable[HiddenGemEntry] | None = None) -> str:
    """Render registry rows as deterministic markdown."""
    rows = tuple(all_hidden_gems() if entries is None else entries)
    lines = [
        "# Hidden-Gem Registry",
        "",
        "Static planning registry. It does not dispatch GPU work and does not carry concrete score claims.",
        "",
        "| key | category | status | integration targets | next patch | evidence | compliance |",
        "|---|---|---|---|---|---|---|",
    ]
    if not rows:
        lines.append("| _none_ | _none_ | _none_ | _none_ | _none_ | _none_ | _none_ |")
    for entry in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{_markdown_cell(entry.key)}`",
                    f"`{_markdown_cell(entry.category)}`",
                    f"`{_markdown_cell(entry.status)}`",
                    _markdown_path_list(entry.integration_targets),
                    _markdown_cell(entry.next_patch),
                    _markdown_path_list(entry.evidence_paths),
                    _markdown_list(entry.contest_compliance_notes),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _markdown_cell(value: str) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _markdown_list(values: Iterable[str]) -> str:
    return "<br>".join(_markdown_cell(value) for value in values)


def _markdown_path_list(values: Iterable[str]) -> str:
    return "<br>".join(f"`{_markdown_cell(value)}`" for value in values)


def _validate_registry(entries: tuple[HiddenGemEntry, ...]) -> None:
    if not entries:
        raise ImportError("hidden_gems: registry must not be empty")

    keys = [entry.key for entry in entries]
    if keys != sorted(keys):
        raise ImportError("hidden_gems: registry entries must be sorted by key")
    if len(keys) != len(set(keys)):
        raise ImportError("hidden_gems: duplicate registry keys")

    for entry in entries:
        _validate_entry(entry)


def _validate_entry(entry: HiddenGemEntry) -> None:
    if not KEY_RE.fullmatch(entry.key):
        raise ImportError(f"hidden_gems: malformed key {entry.key!r}")
    if entry.category not in CATEGORIES:
        raise ImportError(f"hidden_gems: unknown category for {entry.key}: {entry.category!r}")
    if entry.status not in STATUSES:
        raise ImportError(f"hidden_gems: unknown status for {entry.key}: {entry.status!r}")

    for field_name in ("title", "summary", "next_patch"):
        value = getattr(entry, field_name)
        if not value or not value.strip():
            raise ImportError(f"hidden_gems: {entry.key}.{field_name} must be nonempty")
        _validate_ascii_text(entry.key, field_name, value)

    for field_name in ("evidence_paths", "integration_targets", "contest_compliance_notes"):
        values = getattr(entry, field_name)
        if not isinstance(values, tuple) or not values:
            raise ImportError(f"hidden_gems: {entry.key}.{field_name} must be a nonempty tuple")
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ImportError(f"hidden_gems: {entry.key}.{field_name} contains an empty value")
            _validate_ascii_text(entry.key, field_name, value)

    for path in (*entry.evidence_paths, *entry.integration_targets):
        _validate_registry_path(entry.key, path)

    for text in _entry_text_values(entry):
        for pattern in CONCRETE_SCORE_CLAIM_PATTERNS:
            if pattern.search(text):
                raise ImportError(
                    f"hidden_gems: {entry.key} contains a concrete score-claim pattern "
                    f"{pattern.pattern!r}"
                )


def _validate_ascii_text(key: str, field_name: str, value: str) -> None:
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ImportError(f"hidden_gems: {key}.{field_name} must be ASCII") from exc


def _validate_registry_path(key: str, value: str) -> None:
    if value.startswith(("/", "~")) or "\\" in value or "//" in value:
        raise ImportError(f"hidden_gems: {key} path must be repo-relative POSIX: {value!r}")
    parts = PurePosixPath(value).parts
    if not parts or ".." in parts:
        raise ImportError(f"hidden_gems: {key} path must not traverse parents: {value!r}")
    lowered = value.lower()
    if any(lowered.startswith(prefix) for prefix in SENSITIVE_PATH_PREFIXES):
        raise ImportError(f"hidden_gems: {key} path points at provider/private state: {value!r}")
    lowered_parts = {part.lower() for part in parts}
    if lowered_parts & SENSITIVE_PATH_PARTS:
        raise ImportError(f"hidden_gems: {key} path contains a sensitive path part: {value!r}")


def _entry_text_values(entry: HiddenGemEntry) -> tuple[str, ...]:
    return (
        entry.key,
        entry.title,
        entry.category,
        entry.status,
        entry.summary,
        *entry.evidence_paths,
        *entry.integration_targets,
        entry.next_patch,
        *entry.contest_compliance_notes,
    )


_validate_registry(_REGISTRY)


def audit_evidence_paths(
    *,
    repo_root: object | None = None,
) -> dict[str, object]:
    """Verify every evidence_paths entry resolves to an existing file.

    Adversarial review 2026-05-06 (BUG #7, 80% confidence): the registry's
    `_validate_registry_path` checks for absolute paths, parent traversal, and
    sensitive prefixes, but does NOT verify that referenced files actually
    exist on disk. A research file rename or delete leaves the registry with a
    silent dead pointer, and there is no preflight gate equivalent to lane
    registry's Check 90 to catch this.

    This function returns a structured audit (not a strict failure): it walks
    every entry's `evidence_paths` and reports `(gem_key, path, exists)` plus a
    `missing` summary list. Designed to be called warn-only initially; can be
    promoted to STRICT in a future preflight check once the existing registry
    is clean.

    Args:
        repo_root: optional Path-like root to resolve relative paths against.
            Defaults to the repo root inferred from this file's location.

    Returns:
        {
          "total_paths": int,
          "missing_paths": int,
          "missing": [{"gem_key": str, "path": str}, ...],
          "checked_paths": int,
          "schema_version": SCHEMA_VERSION,
        }
    """
    from pathlib import Path

    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    missing: list[dict[str, str]] = []
    total = 0
    checked = 0
    for entry in _REGISTRY:
        for raw_path in entry.evidence_paths:
            total += 1
            # Skip URL-like or non-path tokens; we only audit paths that look
            # like local files. The registry intentionally allows non-path
            # evidence strings (e.g. lane id references); those are not failures.
            if "://" in raw_path or not raw_path or raw_path.startswith("@"):
                continue
            checked += 1
            candidate = (root / raw_path).resolve()
            if not candidate.exists():
                missing.append(
                    {"gem_key": entry.key, "path": raw_path}
                )
    return {
        "total_paths": total,
        "missing_paths": len(missing),
        "checked_paths": checked,
        "missing": missing,
        "schema_version": SCHEMA_VERSION,
    }


__all__ = [
    "CATEGORIES",
    "SCHEMA_VERSION",
    "STATUSES",
    "HiddenGemEntry",
    "all_hidden_gems",
    "audit_evidence_paths",
    "hidden_gem_to_dict",
    "registry_payload",
    "render_markdown",
]
