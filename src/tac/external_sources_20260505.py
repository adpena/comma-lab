"""Contest-faithful design scaffold for 2026-05-05 external-source review.

The registry is intentionally static and import-light. It records implementation
lanes that need exact archive evidence before any score or promotion claim.
"""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = 1

SOURCE_PRIORITIES = frozenset({"primary", "secondary"})

EVIDENCE_GRADES = frozenset(
    {
        "design",
        "scaffold",
        "local_smoke",
        "exact_cuda_required",
    }
)

FORBIDDEN_CONTEST_ACTIONS = frozenset(
    {
        "gpu_dispatch",
        "remote_provider_work",
        "upstream_scorer_patch",
        "uncharged_sidecar",
        "score_claim",
    }
)


@dataclass(frozen=True)
class SourceRecord:
    """One external source used by the dated design ledger."""

    key: str
    title: str
    priority: str
    url: str
    off_the_shelf_status: str
    contest_relevance: tuple[str, ...]
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceGate:
    """A promotion gate that must pass before a lane can make a claim."""

    name: str
    grade: str
    required_artifact: str
    must_record: tuple[str, ...]


@dataclass(frozen=True)
class LaneRecommendation:
    """A contest-faithful implementation lane derived from external sources."""

    key: str
    title: str
    source_keys: tuple[str, ...]
    implementation_target: str
    prototype_scope: str
    contest_safety_notes: tuple[str, ...]
    evidence_gates: tuple[EvidenceGate, ...]
    forbidden_actions: tuple[str, ...] = tuple(sorted(FORBIDDEN_CONTEST_ACTIONS))
    score_claim: bool = False
    remote_dispatch_allowed: bool = False


_SOURCES: tuple[SourceRecord, ...] = (
    SourceRecord(
        key="la_pose",
        title="LA-Pose: Latent Action Pretraining Meets Pose Estimation",
        priority="primary",
        url="https://la-pose.github.io",
        off_the_shelf_status=(
            "No public model weights or implementation were found in the project page, arXiv page, "
            "or source search during this review; use as a design target, not vendored code."
        ),
        contest_relevance=(
            "50-dimensional motion bottleneck is a direct candidate for charged pose sidechannels.",
            "Frozen inverse-dynamics backbone suggests scorer-free inflate with offline training only.",
            "Frame-rate jitter and front-camera driving video match the contest's paired-frame geometry better than generic video priors.",
        ),
        blockers=(
            "Original pretraining used an internal 10.2M driving-video corpus.",
            "No released LA-Pose checkpoint means exact reuse is not currently contest-actionable.",
            "Reverse-motion and medium-curvature regimes need explicit hard-pair guards before dispatch.",
        ),
    ),
    SourceRecord(
        key="ma_gig",
        title="Manifold-Aligned Guided Integrated Gradients",
        priority="secondary",
        url="https://arxiv.org/pdf/2605.02167",
        off_the_shelf_status="MIT implementation exists, but it targets classifier attribution rather than contest codecs.",
        contest_relevance=(
            "Latent-space guided attribution can reduce off-manifold gradient artifacts in PoseNet/SegNet sensitivity maps.",
            "Decoder-Jacobian path framing is useful for HNeRV latent perturbation audits.",
        ),
        blockers=(
            "Requires a trained generative manifold for contest frames.",
            "Attribution maps are diagnostic until exact CUDA archive eval closes the loop.",
        ),
    ),
    SourceRecord(
        key="graph_lottery_ticket",
        title="The Graph Lottery Ticket Hypothesis",
        priority="secondary",
        url="https://arxiv.org/pdf/2312.04762",
        off_the_shelf_status="Algorithmic idea only; simple random-spanning-tree scaffold is easy to reimplement if needed.",
        contest_relevance=(
            "Sparse pair/window graph selection can keep repair atoms connected without densifying charged metadata.",
            "Average-degree targets around 2 to 5 are a useful prior for motion-atom graph budgets.",
        ),
        blockers=(
            "Paper is graph-learning oriented, so contest atom benefit must be measured by component traces.",
            "Random tree sampling must be seeded and manifest-recorded for reproducibility.",
        ),
    ),
    SourceRecord(
        key="manifold_learning_survey",
        title="Manifold learning: what, how, and why",
        priority="secondary",
        url="https://arxiv.org/pdf/2311.03757",
        off_the_shelf_status="Survey and design reference; no direct code dependency.",
        contest_relevance=(
            "Neighborhood scale, tangent-space, and intrinsic-dimension checks are guardrails for LA-Pose latent probes.",
            "Sampling-density bias warnings apply to hard-pair and frame-window latent visualizations.",
        ),
        blockers=(
            "Requires local validation on contest-frame latent clouds.",
            "Embedding plots alone cannot promote or retire a lane.",
        ),
    ),
    SourceRecord(
        key="flowm",
        title="Flow Equivariant World Models",
        priority="secondary",
        url="https://github.com/hlillemark/flowm",
        off_the_shelf_status="MIT code exists, but environments, checkpoints, and datasets are not a drop-in contest runtime.",
        contest_relevance=(
            "Lie-flow equivariance gives a geometry prior for long-horizon memory and camera-motion latent conditioning.",
            "Partially observed dynamics match missing-context risks in mask and renderer prediction.",
        ),
        blockers=(
            "Repository targets synthetic world-model benchmarks, not comma challenge archives.",
            "Any imported model would add charged decoder bytes and dependency risk.",
        ),
    ),
    SourceRecord(
        key="goodfire_vpd",
        title="Interpreting Language Model Parameters",
        priority="secondary",
        url="https://www.goodfire.ai/research/interpreting-lm-parameters",
        off_the_shelf_status="MIT parameter-decomposition code exists, but it is LLM-oriented and too heavy for direct contest inflate.",
        contest_relevance=(
            "Adversarially ablatable subcomponents are a useful audit pattern for no-op-resistant renderer decompositions.",
            "Rank-one parameter decomposition suggests post-training analysis of HNeRV decoder byte importance.",
        ),
        blockers=(
            "Not directly applicable to small image/video renderers without new validation.",
            "Mechanistic explanations are not archive evidence.",
        ),
    ),
    SourceRecord(
        key="architecture_warmup",
        title="Taming Curvature: Architecture Warm-Up for Stable Transformer Training",
        priority="secondary",
        url="https://openreview.net/pdf?id=DuNf2vPTTK",
        off_the_shelf_status="Training recipe only; no contest runtime dependency needed.",
        contest_relevance=(
            "Zero-locked depth growth is a stability guard for any LA-Pose-style transformer pretraining.",
            "Online curvature tracking can classify training aborts before wasting GPU time.",
        ),
        blockers=(
            "Curvature probes are expensive and diagnostic.",
            "No GPU dispatch is allowed from this review.",
        ),
    ),
    SourceRecord(
        key="multiplicative_gaussian_input",
        title="Convergence Analysis of Two-Layer Neural Networks under Gaussian Input Masking",
        priority="secondary",
        url="https://akyrillidis.github.io/aiowls/multiplicative_gaussian_input.html",
        off_the_shelf_status="Reference code exists for validation, but the result is theory-first.",
        contest_relevance=(
            "Multiplicative input noise is a plausible low-risk regularizer for latent-action pretraining on camera/frame-rate variation.",
            "Noise scale must be recorded because it changes the learned target.",
        ),
        blockers=(
            "Two-layer theory does not prove benefit for contest renderers.",
            "Any augmentation must pass no-op and exact CUDA gates.",
        ),
    ),
    SourceRecord(
        key="cauchynet",
        title="CauchyNet: Compact and Data-Efficient Learning using Holomorphic Activation Functions",
        priority="secondary",
        url="https://arxiv.org/pdf/2510.10195",
        off_the_shelf_status="Code is described as forthcoming; treat as architecture inspiration only.",
        contest_relevance=(
            "Compact rational-like activations may fit smooth pose residual functions with fewer charged parameters.",
            "Missing-data interpolation framing is relevant to sparse latent sidechannels.",
        ),
        blockers=(
            "Complex-valued runtime would need a tiny audited implementation before inflate use.",
            "No contest-specific evidence yet.",
        ),
    ),
)


_EXACT_ARCHIVE_GATE = EvidenceGate(
    name="exact_cuda_archive_eval",
    grade="exact_cuda_required",
    required_artifact="experiments/contest_auth_eval.py contest_auth_eval.json",
    must_record=(
        "archive_size_bytes",
        "archive_sha256",
        "runtime_tree_sha256",
        "avg_posenet_dist",
        "avg_segnet_dist",
        "score_recomputed_from_components",
        "n_samples",
    ),
)


_LANES: tuple[LaneRecommendation, ...] = (
    LaneRecommendation(
        key="lapose_posenet_target_distillation",
        title="LA-Pose-style latent-action PoseNet target distillation",
        source_keys=("la_pose", "architecture_warmup", "multiplicative_gaussian_input"),
        implementation_target="src/tac/experiments/train_renderer.py plus a future charged pose-prior payload",
        prototype_scope=(
            "Train an offline inverse-dynamics encoder on contest frames, freeze it, then fit a tiny head to "
            "frozen PoseNet pair targets extracted outside inflate. Export only charged model or sidechannel bytes."
        ),
        contest_safety_notes=(
            "No scorer loads at inflate time.",
            "No uncharged latent sidecars.",
            "Local PoseNet-target agreement is a development diagnostic, not a score.",
        ),
        evidence_gates=(
            EvidenceGate(
                name="offline_target_manifest",
                grade="scaffold",
                required_artifact="manifest with source frames, seeds, target extraction command, tensor hashes",
                must_record=("pair_count", "frame_stride_policy", "target_sha256", "encoder_config_sha256"),
            ),
            EvidenceGate(
                name="charged_payload_manifest",
                grade="local_smoke",
                required_artifact="archive manifest proving every latent/action/model byte is charged",
                must_record=("member_names", "member_sha256s", "charged_bytes", "inflate_runtime_files"),
            ),
            _EXACT_ARCHIVE_GATE,
        ),
    ),
    LaneRecommendation(
        key="lapose_hnerv_latent_conditioning",
        title="LA-Pose bottleneck conditioning for HNeRV latent or residual streams",
        source_keys=("la_pose", "flowm", "manifold_learning_survey"),
        implementation_target="future HNeRV renderer conditioning path with quantized latent-action stream",
        prototype_scope=(
            "Encode 16-frame windows into compact motion tokens, quantize or entropy-code the tokens, and use "
            "them only as charged conditioning for an HNeRV/residual decoder."
        ),
        contest_safety_notes=(
            "Conditioning stream must be byte-accounted and deterministic.",
            "Ablations must prove the stream changes decoded frames and is not metadata-only.",
            "Runtime must use canonical loaders for packed renderer formats.",
        ),
        evidence_gates=(
            EvidenceGate(
                name="latent_noop_guard",
                grade="local_smoke",
                required_artifact="decode-diff report for zeroed, shuffled, and original latent streams",
                must_record=("decoded_frame_sha256s", "latent_stream_sha256", "changed_pixels", "seed"),
            ),
            EvidenceGate(
                name="entropy_budget_report",
                grade="local_smoke",
                required_artifact="quantized latent byte anatomy and entropy estimate",
                must_record=("raw_bytes", "coded_bytes", "histogram_overhead", "decoder_bytes"),
            ),
            _EXACT_ARCHIVE_GATE,
        ),
    ),
    LaneRecommendation(
        key="lapose_motion_atom_sparsifier",
        title="Motion-manifold sparse atom planner for pose-sensitive repair",
        source_keys=("la_pose", "ma_gig", "graph_lottery_ticket", "manifold_learning_survey", "goodfire_vpd"),
        implementation_target="future atom planner that selects connected sparse frame/pair repair windows",
        prototype_scope=(
            "Build a deterministic graph over pair-window latent actions, sparsify it with seeded spanning-tree "
            "tickets, and rank atoms by CUDA component traces plus manifold-aligned attribution diagnostics."
        ),
        contest_safety_notes=(
            "Graph and attribution outputs are planning signals only.",
            "Selection policies must record seeds and avoid stochastic hidden state.",
            "Pose-sensitive atoms require source/candidate archive SHA matching before dispatch readiness.",
        ),
        evidence_gates=(
            EvidenceGate(
                name="deterministic_sparse_graph_manifest",
                grade="scaffold",
                required_artifact="seeded graph manifest with source pair ids and selected edges",
                must_record=("node_count", "edge_count", "average_degree", "rng_seed", "input_sha256"),
            ),
            EvidenceGate(
                name="component_trace_crosscheck",
                grade="local_smoke",
                required_artifact="CUDA component trace cross-check against exact eval components",
                must_record=("trace_archive_sha256", "device", "pair_ids", "component_deltas"),
            ),
            _EXACT_ARCHIVE_GATE,
        ),
    ),
)


def all_sources() -> tuple[SourceRecord, ...]:
    """Return source records in stable priority order."""

    return _SOURCES


def all_lanes() -> tuple[LaneRecommendation, ...]:
    """Return lane recommendations in ranked implementation order."""

    return _LANES


def source_by_key(key: str) -> SourceRecord:
    """Return a source record or raise KeyError for unknown keys."""

    for source in _SOURCES:
        if source.key == key:
            return source
    raise KeyError(key)


def validate_lane(lane: LaneRecommendation) -> None:
    """Fail closed if a recommendation violates contest-safety constraints."""

    source_keys = {source.key for source in _SOURCES}
    missing_sources = set(lane.source_keys) - source_keys
    if missing_sources:
        raise ValueError(f"{lane.key}: unknown source keys {sorted(missing_sources)}")
    if lane.score_claim:
        raise ValueError(f"{lane.key}: scaffold must not claim a score")
    if lane.remote_dispatch_allowed:
        raise ValueError(f"{lane.key}: scaffold must not allow remote dispatch")
    missing_forbidden = FORBIDDEN_CONTEST_ACTIONS - set(lane.forbidden_actions)
    if missing_forbidden:
        raise ValueError(f"{lane.key}: missing forbidden actions {sorted(missing_forbidden)}")
    gate_names = {gate.name for gate in lane.evidence_gates}
    if "exact_cuda_archive_eval" not in gate_names:
        raise ValueError(f"{lane.key}: missing exact CUDA archive-eval gate")
    for gate in lane.evidence_gates:
        if gate.grade not in EVIDENCE_GRADES:
            raise ValueError(f"{lane.key}: invalid evidence grade {gate.grade}")
        if not gate.required_artifact or not gate.must_record:
            raise ValueError(f"{lane.key}: incomplete gate {gate.name}")


def registry_payload() -> dict[str, object]:
    """Serialize the static registry for operator review."""

    for lane in _LANES:
        validate_lane(lane)
    return {
        "schema_version": SCHEMA_VERSION,
        "registry": "external_sources_20260505",
        "sources": [_source_to_dict(source) for source in _SOURCES],
        "lanes": [_lane_to_dict(lane) for lane in _LANES],
    }


def _source_to_dict(source: SourceRecord) -> dict[str, object]:
    return {
        "key": source.key,
        "title": source.title,
        "priority": source.priority,
        "url": source.url,
        "off_the_shelf_status": source.off_the_shelf_status,
        "contest_relevance": list(source.contest_relevance),
        "blockers": list(source.blockers),
    }


def _gate_to_dict(gate: EvidenceGate) -> dict[str, object]:
    return {
        "name": gate.name,
        "grade": gate.grade,
        "required_artifact": gate.required_artifact,
        "must_record": list(gate.must_record),
    }


def _lane_to_dict(lane: LaneRecommendation) -> dict[str, object]:
    return {
        "key": lane.key,
        "title": lane.title,
        "source_keys": list(lane.source_keys),
        "implementation_target": lane.implementation_target,
        "prototype_scope": lane.prototype_scope,
        "contest_safety_notes": list(lane.contest_safety_notes),
        "evidence_gates": [_gate_to_dict(gate) for gate in lane.evidence_gates],
        "forbidden_actions": list(lane.forbidden_actions),
        "score_claim": lane.score_claim,
        "remote_dispatch_allowed": lane.remote_dispatch_allowed,
    }
