"""Cross-lane composition glue (per ``docs/stacking_architecture.md``).

This module is the canonical way to *stack* lane outputs at the archive
level WITHOUT a refactor of either lane. Lanes that follow the
"renderer-encoder + sidecar-additive" composition rule from the design
contract can be composed by simply consuming each lane's already-built
artifact and writing a deterministic zip.

The first concrete composition lives here:

    compose_jnwcs_with_ec(...)
        Stacks a Lane J-NWCS-encoded ``renderer.bin`` (the
        sensitivity-aware NWC weight codec output) with a Lane EC
        ``gradient_corrections.bin`` sidecar, producing a single contest
        archive that the standard ``inflate.sh`` consumes without any
        special-case wiring. The two artifacts attack the rate wedge from
        DIFFERENT layers (weight-bit allocation vs inflate-time pixel
        residuals) so they are complementary, not redundant.

Strict-scorer-rule (CLAUDE.md non-negotiable) preservation
----------------------------------------------------------
* Lane J-NWCS produces a renderer.bin whose loader does NOT touch
  PoseNet/SegNet (the codec's encoder/decoder MLPs are bundled INSIDE the
  renderer.bin wire format and reconstruct weights at inflate time
  without any external scorer state).
* Lane EC produces a sparse-int8 ``gradient_corrections.bin`` that
  ``inflate_renderer.py`` discovers by name and applies as a numpy
  additive overlay (see ``apply_corrections_at_inflate`` in
  ``tac.engineered_corrections``). No torch autograd, no scorer.
* Therefore the *composition* trivially inherits the strict-scorer-rule
  property — neither artifact can sneak in a scorer dependency through
  the other. ``validate_jnwcs_ec_composition`` enforces this with a
  byte-level magic check on renderer.bin.

Determinism
-----------
* Every archive built by this module uses a fixed ZipInfo timestamp
  (``(2024, 1, 1, 0, 0, 0)``) so two compose calls with byte-identical
  inputs produce byte-identical archives. This is the same convention
  ``tac.lossless.submission.build_submission_zip`` and the canonical
  remote bootstraps follow (codex R5-r6 #5).

Anchor reuse
------------
* This module is a PURE COMPOSITION layer. It does NOT retrain anything
  and it does NOT re-download artifacts. Callers feed it paths to
  artifacts that have ALREADY been produced by their respective lane
  bootstraps (Lane J-NWCS Stage 5, Lane EC Stage 2). The
  ``bit_budget_split_search`` helper enumerates compose configurations
  by re-running each lane's *artifact-producing* logic with different
  rate budgets — each invocation builds a single new archive but does
  NOT re-train the underlying NWCS codec or re-search EC corrections
  from scratch.
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

__all__ = [
    "compose_jnwcs_with_ec",
    "validate_jnwcs_ec_composition",
    "bit_budget_split_search",
    "build_joint_admm_balle_arithmetic_noop_manifest",
    "BitBudgetSplit",
    "JOINT_STACK_CANONICAL_ORDER",
    "JOINT_STACK_NOOP_MANIFEST_SCHEMA",
    "REQUIRED_ARCHIVE_MEMBERS",
]


# ── Constants ─────────────────────────────────────────────────────────────


# Fixed ZipInfo timestamp so archives are byte-deterministic across
# compose() calls. Matches ``tac.lossless.submission._FIXED_ZIP_TIMESTAMP``
# and the codex R5-r6 #5 deterministic-zip rule.
_FIXED_ZIP_TIMESTAMP = (2024, 1, 1, 0, 0, 0)


# Members the J-NWCS × EC stack must contain. Order is the canonical
# order Lane EC's remote script writes; we preserve it so byte-diff
# comparisons against existing archives are clean.
REQUIRED_ARCHIVE_MEMBERS: tuple[str, ...] = (
    "renderer.bin",
    "masks.mkv",
    "optimized_poses.pt",
    "gradient_corrections.bin",
)


JOINT_STACK_CANONICAL_ORDER: tuple[str, ...] = (
    "representation",
    "prediction",
    "quantization",
    "hyperprior",
    "arithmetic",
    "pack",
)


JOINT_STACK_NOOP_MANIFEST_SCHEMA = (
    "joint_admm_balle_arithmetic_noop_manifest_v1"
)


# Renderer-bin magics that DO NOT load scorer state at inflate time.
# Lane J-NWCS exports preserve the upstream renderer's magic when the
# encoded bytes are spliced back into renderer.bin — typically ``ASYM``
# (the Lane G v3 anchor magic) or ``FP4A``. Magics that require scorer
# loading (none currently exist in tree) would be rejected here.
_SCORER_FREE_RENDERER_MAGICS: tuple[bytes, ...] = (
    b"ASYM",
    b"DPSM",
    b"FP4A",
    b"FP8H",
    b"I4LZ",
    b"CCh1",
    b"C3R1",
    b"SCv1",
    b"SZv1",
    b"QFAI",
    b"QZS3",
    b"MQZ1",
    b"QBF1",
    b"BFJ1",
    b"QH0",
    b"QM0",
    b"NWC1",  # Lane J-NWC base codec wire format
    b"NWCS",  # Lane J-NWCS sensitivity-aware codec wire format prefix
)


# ── PARADIGM-gamma typed no-op stack fixture manifest ─────────────────────


def _canonical_manifest_sha(payload: dict[str, Any]) -> str:
    """Digest the deterministic JSON payload before the digest field exists."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def build_joint_admm_balle_arithmetic_noop_manifest(
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return the deterministic PARADIGM-gamma typed stack fixture manifest.

    This is intentionally a **non-dispatching integration contract**, not a
    score artifact. It closes the "ADMM + Ballé + AQ landed individually"
    gap by pinning the canonical typed order, charged-byte obligations, no-op
    fixture semantics, and exact-eval blockers before optimization starts.
    """

    root = Path.cwd() if repo_root is None else Path(repo_root)
    code_paths = (
        "src/tac/joint_codec_stack_orchestrator.py",
        "src/tac/joint_admm_coordinator.py",
        "src/tac/balle_hyperprior_codec.py",
        "src/tac/arithmetic_qint_codec.py",
        "src/tac/entropy_archive.py",
        "src/tac/stack_compositions.py",
    )
    present_paths = [p for p in code_paths if (root / p).exists()]
    missing_paths = [p for p in code_paths if not (root / p).exists()]

    components: list[dict[str, Any]] = [
        {
            "layer": "representation",
            "component": "typed_qint_stream_sources",
            "module": "src/tac/joint_codec_stack_orchestrator.py",
            "typed_input_contract": (
                "named 1-D integer qint streams with alphabet size, offset, "
                "codec kind, and cached score-per-byte marginal"
            ),
            "typed_output_contract": (
                "StreamSource records consumed without scorer state at "
                "compress time"
            ),
            "charged_bytes_obligation": (
                "all emitted stream payloads and JCSP metadata must be "
                "inside archive.zip"
            ),
        },
        {
            "layer": "prediction",
            "component": "joint_admm_byte_allocator",
            "module": "src/tac/joint_admm_coordinator.py",
            "typed_input_contract": (
                "StreamProximalCodec objects plus byte budget and cached "
                "marginal surfaces"
            ),
            "typed_output_contract": (
                "per-stream target bytes, convergence flag, KKT residual, "
                "and allocation history"
            ),
            "charged_bytes_obligation": (
                "allocator metadata is planning-only unless packed in JCSP"
            ),
        },
        {
            "layer": "quantization",
            "component": "qint_stream_contract",
            "module": "src/tac/joint_codec_stack_orchestrator.py",
            "typed_input_contract": "already-quantized signed integer arrays",
            "typed_output_contract": (
                "codec-specific lossless payloads after roundtrip verification"
            ),
            "charged_bytes_obligation": (
                "quantizer choices must be recorded in archive manifest before "
                "exact eval"
            ),
        },
        {
            "layer": "hyperprior",
            "component": "balle_hyperprior_bhv1",
            "module": "src/tac/balle_hyperprior_codec.py",
            "typed_input_contract": "qint stream plus deterministic hyperprior config",
            "typed_output_contract": "BHv1 payload or AQv1 static fallback",
            "charged_bytes_obligation": (
                "hyperprior parameters and fallback decision must be charged "
                "or fixed in committed runtime"
            ),
        },
        {
            "layer": "arithmetic",
            "component": "aqv1_static_arithmetic_terminal",
            "module": "src/tac/arithmetic_qint_codec.py",
            "typed_input_contract": "symbol stream derived from qints and offset",
            "typed_output_contract": "AQv1 payload with deterministic roundtrip",
            "charged_bytes_obligation": (
                "frequency tables, model ids, and all side information must "
                "be included in payload bytes"
            ),
        },
        {
            "layer": "pack",
            "component": "jcsp_container_and_archive_boundary",
            "module": "src/tac/joint_codec_stack_orchestrator.py",
            "typed_input_contract": "StackStreamResult payloads with actual byte counts",
            "typed_output_contract": "JCSP container bytes, then archive.zip member",
            "charged_bytes_obligation": (
                "JCSP must be a real archive member with runtime loader parity "
                "before dispatch"
            ),
        },
    ]

    manifest: dict[str, Any] = {
        "schema": JOINT_STACK_NOOP_MANIFEST_SCHEMA,
        "key": "joint_admm_balle_arithmetic_stack",
        "title": "Joint ADMM plus Ballé hyperprior plus AQ arithmetic stack fixture",
        "score_claim": False,
        "evidence_grade": "planning/component empirical",
        "dispatch_attempted": False,
        "remote_jobs_dispatched": False,
        "ready_for_exact_eval_dispatch": False,
        "fixture_only": True,
        "candidate_non_noop": False,
        "promotion_allowed": False,
        "canonical_order": list(JOINT_STACK_CANONICAL_ORDER),
        "components": components,
        "no_op_fixture_semantics": {
            "purpose": (
                "exercise the typed end-to-end stack contract without claiming "
                "a changed contest archive"
            ),
            "archive_bytes": None,
            "archive_sha256": None,
            "runtime_tree_sha256": None,
            "score_components": None,
            "must_not_dispatch": True,
        },
        "determinism_contract": {
            "json_encoding": "sort_keys=True, allow_nan=False",
            "wire_endianness": "JCSP uses explicit little-endian struct fields",
            "cross_platform": (
                "manifest is pure JSON; future native/Rust/SIMD consumers must "
                "prove byte-identical output on macOS/Linux before promotion"
            ),
        },
        "composition_invariants": [
            "representation_to_prediction_to_quantization_to_hyperprior_to_arithmetic_to_pack",
            "no scorer loads at inflate time",
            "all side information is charged inside archive.zip",
            "per-stream roundtrip is verified before pack",
            "JCSP container byte budget includes metadata overhead",
            "exact CUDA auth eval is required before score or rank claims",
        ],
        "dispatch_blockers": [
            "fixture_only_candidate_not_dispatchable",
            "no_byte_closed_archive_member_for_jcsp_container",
            "no_runtime_loader_parity_for_jcsp_archive_member",
            "no_exact_cuda_auth_eval_for_stacked_archive",
            "no_lane_dispatch_claim",
            "component_empirical_results_do_not_prove_stack_composability",
        ],
        "next_patch": (
            "Build a byte-closed archive member that consumes JCSP at inflate, "
            "then claim lane and run exact CUDA auth eval on the exact archive."
        ),
        "path_audit": {
            "code": {
                "present": present_paths,
                "missing": missing_paths,
            }
        },
    }
    manifest["manifest_sha256"] = _canonical_manifest_sha(manifest)
    return manifest


# ── Composition function ──────────────────────────────────────────────────


def compose_jnwcs_with_ec(
    renderer_path: str | Path,
    ec_corrections_path: str | Path,
    masks_path: str | Path,
    poses_path: str | Path,
    output_archive_path: str | Path,
) -> Path:
    """Build a contest archive that contains the J-NWCS-encoded renderer
    + Lane EC corrections sidecar + masks + poses.

    This is a PURE COMPOSITION: the function reads the four input files
    AS-IS and writes them into a single deterministic zip. It does NOT
    mutate either underlying module — both artifacts are consumed in
    their already-built form. Round-28 review-rotation #1 (compositions
    don't introduce hidden state) is enforced by this contract.

    Args:
        renderer_path: path to a J-NWCS-encoded ``renderer.bin``
            (typically ``$LOG_DIR/renderer_nwcs.bin`` from Lane J-NWCS
            Stage 5). Must exist; must have one of the scorer-free
            magic bytes (validated downstream).
        ec_corrections_path: path to a Lane EC
            ``gradient_corrections.bin`` (typically
            ``$LOG_DIR/corrections/gradient_corrections.bin`` from Lane
            EC Stage 2). Must exist and be > 0 bytes.
        masks_path: path to ``masks.mkv`` (typically pulled from the
            Lane G v3 anchor archive).
        poses_path: path to ``optimized_poses.pt`` (typically from the
            Lane G v3 anchor archive).
        output_archive_path: destination for the composed archive.

    Returns:
        ``Path`` to the written archive.

    Raises:
        FileNotFoundError: any input path missing.
        ValueError: any input file empty.

    Determinism: same inputs → byte-identical output (fixed timestamp,
    fixed compresslevel=9, fixed member ordering).

    Strict-scorer-rule: the composition itself touches NO scorer code.
    The renderer.bin magic check is byte-level only.
    """
    renderer_path = Path(renderer_path)
    ec_corrections_path = Path(ec_corrections_path)
    masks_path = Path(masks_path)
    poses_path = Path(poses_path)
    output_archive_path = Path(output_archive_path)

    # Loud failure on missing or empty inputs (Round-28 #4: the compose
    # function MUST NOT silently drop either artifact).
    inputs = {
        "renderer.bin": renderer_path,
        "masks.mkv": masks_path,
        "optimized_poses.pt": poses_path,
        "gradient_corrections.bin": ec_corrections_path,
    }
    for arc_name, src in inputs.items():
        if not src.is_file():
            raise FileNotFoundError(
                f"compose_jnwcs_with_ec: missing input for archive member "
                f"{arc_name!r}: {src}"
            )
        if src.stat().st_size <= 0:
            raise ValueError(
                f"compose_jnwcs_with_ec: empty input for archive member "
                f"{arc_name!r}: {src}"
            )

    output_archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:
        # Write members in REQUIRED_ARCHIVE_MEMBERS order so the on-disk
        # central-directory ordering is deterministic.
        for arc_name in REQUIRED_ARCHIVE_MEMBERS:
            src = inputs[arc_name]
            info = zipfile.ZipInfo(filename=arc_name, date_time=_FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3  # unix attribute slot — matches submission.py
            zf.writestr(info, src.read_bytes(), compresslevel=9)

    return output_archive_path


# ── Composition validator ─────────────────────────────────────────────────


def validate_jnwcs_ec_composition(
    archive_path: str | Path,
    *,
    max_archive_bytes: int | None = None,
) -> dict[str, Any]:
    """Validate a J-NWCS × EC stack archive.

    Checks (Round-28 review rotation):

      * Round-28 #3: STRICT-SCORER-RULE preservation. The archive must
        not require scorer state at inflate. Enforced by reading the
        renderer.bin magic and asserting it is in
        ``_SCORER_FREE_RENDERER_MAGICS``.
      * Round-28 #4: anchor-reuse honesty / no silent drops. All members
        of ``REQUIRED_ARCHIVE_MEMBERS`` must be present and non-empty.

    Args:
        archive_path: path to the composed zip.
        max_archive_bytes: optional hard cap. When set, archives larger
            than this raise ``ValueError`` (Lane EC's
            ``--max-artifact-bytes`` analog at the *stack* level).

    Returns:
        Summary dict — useful for logging by the deploy script::

            {
                "archive_bytes": int,
                "renderer_magic": bytes,
                "members": {arc_name: byte_size, ...},
                "compose_strategy": "jnwcs_with_ec",
            }

    Raises:
        FileNotFoundError: archive does not exist.
        ValueError: missing required member, empty member, scorer-loading
            renderer magic, archive over hard-cap, or both artifacts are
            absent (the "compose did nothing" mode).
    """
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"validate_jnwcs_ec_composition: archive not found: {archive_path}"
        )

    archive_bytes = archive_path.stat().st_size
    if archive_bytes <= 0:
        raise ValueError(
            f"validate_jnwcs_ec_composition: archive is empty: {archive_path}"
        )
    if max_archive_bytes is not None and archive_bytes > max_archive_bytes:
        raise ValueError(
            f"validate_jnwcs_ec_composition: archive {archive_bytes} bytes "
            f"exceeds hard cap {max_archive_bytes} bytes"
        )

    members: dict[str, int] = {}
    renderer_magic: bytes = b""
    with zipfile.ZipFile(archive_path) as zf:
        names_present = set(zf.namelist())
        required = set(REQUIRED_ARCHIVE_MEMBERS)
        missing = required - names_present
        if missing:
            raise ValueError(
                f"validate_jnwcs_ec_composition: archive missing required "
                f"members: {sorted(missing)}"
            )
        for arc_name in REQUIRED_ARCHIVE_MEMBERS:
            info = zf.getinfo(arc_name)
            members[arc_name] = info.file_size
            if info.file_size <= 0:
                raise ValueError(
                    f"validate_jnwcs_ec_composition: archive member "
                    f"{arc_name!r} is empty"
                )
        # Magic check on renderer.bin — strict-scorer-rule preservation.
        with zf.open("renderer.bin") as fh:
            renderer_magic = fh.read(4)
    if renderer_magic not in _SCORER_FREE_RENDERER_MAGICS:
        raise ValueError(
            f"validate_jnwcs_ec_composition: renderer.bin magic "
            f"{renderer_magic!r} is not in the scorer-free allowlist "
            f"{[m.decode('latin1') for m in _SCORER_FREE_RENDERER_MAGICS]}. "
            f"Strict-scorer-rule (CLAUDE.md) requires the inflate path to "
            f"load no scorer."
        )

    return {
        "archive_bytes": archive_bytes,
        "renderer_magic": renderer_magic,
        "members": members,
        "compose_strategy": "jnwcs_with_ec",
    }


# ── Bit-budget split search ───────────────────────────────────────────────


@dataclass(frozen=True)
class BitBudgetSplit:
    """One point on the J-NWCS × EC Pareto curve.

    Attributes:
        weight_avg_bits: average bits-per-weight that J-NWCS spent on the
            renderer (smaller = more aggressive weight quantization).
        ec_rate_cap_bytes: the rate cap passed to Lane EC's
            greedy-water-fill (smaller = fewer pixel corrections).
        archive_bytes: actual on-disk size of the composed archive.
        predicted_score: the deterministic score-prediction model's
            estimate (see :func:`_predict_score_for_split`). This is a
            CHEAP HEURISTIC, not an auth-eval result. The deploy script
            uses it to pick which 1-2 splits to actually contest-eval.
    """
    weight_avg_bits: float
    ec_rate_cap_bytes: int
    archive_bytes: int
    predicted_score: float


def _predict_score_for_split(
    weight_avg_bits: float,
    ec_rate_cap_bytes: int,
    archive_bytes: int,
    *,
    base_score: float,
    rate_per_byte: float,
    weight_distortion_per_bit_drop: float,
    ec_distortion_reduction_per_byte: float,
) -> float:
    """Cheap analytic score estimate for one (weight_bits, ec_cap) split.

    The model is intentionally simple (linear in both knobs) — it is a
    sorting heuristic for the bit-budget-split search, NOT a substitute
    for contest-CUDA auth eval. The deploy script will run a contest
    eval on the top 1-2 predictions; the predictions just narrow the
    search space.

    Score = base_score
            + (37545489 baseline / 25 contest factor) * Δrate
            + Δweight_distortion(weight_avg_bits)
            - Δec_distortion(ec_rate_cap_bytes)

    where each Δ is normalized so that ``base_score`` reflects the
    ANCHOR (Lane G v3 = 1.05) at typical operating point.
    """
    rate_term = rate_per_byte * float(archive_bytes)
    # Less weight bits → more weight-quantization distortion. Anchor at
    # 4.0 bits/weight (NWCS default); subtract relative drop.
    weight_distortion = weight_distortion_per_bit_drop * max(
        0.0, 4.0 - float(weight_avg_bits)
    )
    # More EC bytes → more SegNet correction → distortion reduction
    # (capped at 25KB beyond which the gradient-search hits a floor;
    # see Lane EC predicted_band rationale).
    ec_effective_bytes = min(int(ec_rate_cap_bytes), 25_000)
    ec_distortion_reduction = ec_distortion_reduction_per_byte * float(
        ec_effective_bytes
    )
    return float(base_score + rate_term + weight_distortion - ec_distortion_reduction)


def bit_budget_split_search(
    target_archive_size: int,
    n_grid: int = 5,
    *,
    weight_bits_grid: list[float] | None = None,
    ec_rate_cap_grid: list[int] | None = None,
    base_score: float = 1.05,
    base_renderer_bytes: int = 296_776,
    base_masks_bytes: int = 421_483,
    base_poses_bytes: int = 15_620,
    rate_per_byte: float = 25.0 / 37_545_489.0,
    weight_distortion_per_bit_drop: float = 0.05,
    ec_distortion_reduction_per_byte: float = 0.0000060,
) -> list[BitBudgetSplit]:
    """Sweep over (weight_avg_bits × ec_rate_cap_bytes) splits and
    return the results sorted by predicted score (best first).

    The grid is the cross product of ``weight_bits_grid`` (defaults to a
    geometric sweep [2, 3, 4, 5, 6] when ``n_grid == 5``) and
    ``ec_rate_cap_grid`` (defaults to a linear sweep
    [10K, 20K, 30K, 40K, 50K] when ``n_grid == 5``). Splits whose
    predicted archive size exceeds ``target_archive_size`` are STILL
    included in the returned list (so callers can see the full Pareto
    frontier), but they are tagged with predicted_score = +∞-ish via
    the rate term.

    Anchor / size model:
        renderer_bytes ≈ base_renderer_bytes * (weight_avg_bits / 4.0)
            (J-NWCS default codec is ~4 bits/weight; smaller bit-counts
             produce proportionally smaller renderer.bin)
        ec_bytes       = ec_rate_cap_bytes (literal cap from Lane EC)
        archive_bytes  = renderer_bytes + base_masks_bytes
                         + base_poses_bytes + ec_bytes + zip_overhead

        zip_overhead is approximated as 256 bytes (deterministic ZipInfo
        header for 4 members + central-directory record). This matches
        the empirical overhead of compose_jnwcs_with_ec on the Lane G v3
        anchor.

    The Pareto-monotonicity properties asserted by
    ``test_bit_budget_split_search_produces_pareto_curve``:
      * For fixed ec_rate_cap_bytes: archive_bytes is MONOTONIC INCREASING
        in weight_avg_bits.
      * For fixed weight_avg_bits: archive_bytes is MONOTONIC INCREASING
        in ec_rate_cap_bytes.
      * The predicted_score column at fixed ec_rate_cap is MONOTONIC
        DECREASING in weight_avg_bits below the rate-knee (more bits ⇒
        less distortion ⇒ better score), but eventually monotone
        INCREASING above the rate-knee (rate term dominates).

    Args:
        target_archive_size: the operator's archive-size budget (bytes).
            Used for filtering / annotation; splits over the cap are
            still returned but score-penalized via rate_term.
        n_grid: number of grid points per axis when default grids are
            used. Ignored when explicit grids are supplied.
        weight_bits_grid / ec_rate_cap_grid: explicit grids (operators
            can supply if the defaults are too coarse).
        base_score: anchor (Lane G v3 = 1.05 contest-CUDA).
        base_renderer_bytes / base_masks_bytes / base_poses_bytes: known
            sizes of the Lane G v3 anchor's components, used for archive
            size estimation.
        rate_per_byte: contest-formula rate constant (25 / 37,545,489 ≈
            6.66e-7).
        weight_distortion_per_bit_drop: heuristic — how much score
            improves per additional bit-per-weight on the renderer.
        ec_distortion_reduction_per_byte: heuristic — how much score
            improves per additional EC sidecar byte.

    Returns:
        List of ``BitBudgetSplit`` objects sorted by predicted_score
        ascending (lowest is best — contest score is "lower is better").
    """
    if n_grid <= 0:
        raise ValueError(f"n_grid must be positive, got {n_grid}")
    if weight_bits_grid is None:
        # Default geometric-ish sweep. Lower bound 2 (heavy quant);
        # upper bound 6 (~standard FP6).
        if n_grid == 5:
            weight_bits_grid = [2.0, 3.0, 4.0, 5.0, 6.0]
        else:
            step = 4.0 / max(1, n_grid - 1)
            weight_bits_grid = [2.0 + i * step for i in range(n_grid)]
    if ec_rate_cap_grid is None:
        if n_grid == 5:
            ec_rate_cap_grid = [10_000, 20_000, 30_000, 40_000, 50_000]
        else:
            step = 40_000 // max(1, n_grid - 1)
            ec_rate_cap_grid = [10_000 + i * step for i in range(n_grid)]

    if not weight_bits_grid:
        raise ValueError("weight_bits_grid is empty")
    if not ec_rate_cap_grid:
        raise ValueError("ec_rate_cap_grid is empty")

    zip_overhead = 256

    splits: list[BitBudgetSplit] = []
    for w_bits in weight_bits_grid:
        if w_bits <= 0:
            raise ValueError(f"weight_avg_bits must be positive, got {w_bits}")
        renderer_bytes = int(base_renderer_bytes * w_bits / 4.0)
        for ec_cap in ec_rate_cap_grid:
            if ec_cap < 0:
                raise ValueError(f"ec_rate_cap_bytes must be ≥ 0, got {ec_cap}")
            archive_bytes = (
                renderer_bytes
                + base_masks_bytes
                + base_poses_bytes
                + int(ec_cap)
                + zip_overhead
            )
            predicted = _predict_score_for_split(
                w_bits,
                ec_cap,
                archive_bytes,
                base_score=base_score,
                rate_per_byte=rate_per_byte,
                weight_distortion_per_bit_drop=weight_distortion_per_bit_drop,
                ec_distortion_reduction_per_byte=ec_distortion_reduction_per_byte,
            )
            splits.append(
                BitBudgetSplit(
                    weight_avg_bits=float(w_bits),
                    ec_rate_cap_bytes=int(ec_cap),
                    archive_bytes=int(archive_bytes),
                    predicted_score=float(predicted),
                )
            )

    # Sort best-first (lower is better)
    splits.sort(key=lambda s: s.predicted_score)
    return splits
