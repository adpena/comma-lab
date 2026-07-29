#!/usr/bin/env python
"""One-pair stale rehearsal of the terminal six-equation pose GN mechanism.

The exact frozen composed camera bytes and banked six-value target are loaded
from SSD custody.  The receipt runs both a deterministic mechanism canary and
the pinned frozen PoseNet on one pair.  Neither mode can support a score or
promotion claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from tac.optimization.terminal_pose_gn import (
    STALE_POSE_REHEARSAL_AUTHORITY_MARKER,
    CandidateArtifactScope,
    PoseAuthorityMode,
    PoseJointEvaluation,
    TerminalPoseCandidateArtifact,
    TerminalPoseGNConfig,
    TerminalPosePacketV1,
    parse_terminal_pose_packet,
    serialize_terminal_pose_packet,
    solve_terminal_pose_gn,
)

DEFAULT_FRAME_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_ct1_campaign_telemetry_encode_20260725/"
    "e5a_runtime/output_identity/"
    "2a2c0367150f8c8c0953dfb5c1485e238bbc9995c37385e149e52ae22f506241"
)
DEFAULT_PAIRS_RAW = DEFAULT_FRAME_ROOT / "pairs_0000_0032.raw"
DEFAULT_PAIRS_SIDECAR = DEFAULT_FRAME_ROOT / "pairs_0000_0032.json"
DEFAULT_TARGETS = Path(
    "/Volumes/VertigoDataTier/pact/"
    "ddm_ms4_metric_producers_and_measurement_20260724T042005Z/"
    "pose_metric_n600_batch32.json"
)
DEFAULT_UPSTREAM_ROOT = Path("/Volumes/VertigoDataTier/pact/molab_witness_machine_upstream_20260709")
PAIR_SHAPE = (2, 874, 1164, 3)
CHUNK_PAIRS = 32
SEED = 20260728
BASIS_SELECTOR = "eg1_generic_low_frequency_six_v1"
AMPLITUDE_Q8 = 512


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_posenet(upstream_root: Path):
    """Load the pinned upstream CPU PoseNet without any video dataset."""

    import torch
    from safetensors.torch import load_file

    modules_path = upstream_root / "modules.py"
    model_path = upstream_root / "models/posenet.safetensors"
    if not modules_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("pinned upstream PoseNet custody is unavailable")
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))
    import modules as upstream_modules

    if Path(upstream_modules.__file__).resolve() != modules_path.resolve():
        raise RuntimeError("frozen PoseNet imported non-custodied modules.py")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    posenet = upstream_modules.PoseNet().eval().to("cpu")
    posenet.load_state_dict(load_file(str(model_path), device="cpu"))
    for parameter in posenet.parameters():
        parameter.requires_grad = False
    return posenet, model_path


def _frozen_posenet6(posenet, pair: np.ndarray) -> np.ndarray:
    import torch

    pair_tensor = torch.from_numpy(np.asarray(pair)[None]).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        output = posenet(posenet.preprocess_input(pair_tensor))
        pose = output["pose"] if isinstance(output, dict) else output
    result = pose[0, :6].cpu().numpy().astype(np.float64)
    if result.shape != (6,) or not np.all(np.isfinite(result)):
        raise RuntimeError("frozen PoseNet returned malformed pose6")
    return result


def _render_basis(seed: int, basis_selector: str, shape: tuple[int, int, int]) -> np.ndarray:
    if seed != SEED or basis_selector != BASIS_SELECTOR:
        raise RuntimeError("terminal pose basis packet key differs")
    height, width, channels = shape
    if channels != 3:
        raise RuntimeError("terminal pose rehearsal expects RGB")
    x = np.cos(2.0 * np.pi * (np.arange(width, dtype=np.float64) + 0.5) / width)
    y = np.cos(2.0 * np.pi * (np.arange(height, dtype=np.float64) + 0.5) / height)
    fields = np.zeros((6, height, width, channels), dtype=np.float32)
    for channel in range(3):
        fields[channel, :, :, channel] = x[None, :]
        fields[channel + 3, :, :, channel] = y[:, None]
    return fields


def rehearse(
    pairs_raw: Path,
    pairs_sidecar: Path,
    targets_path: Path,
    *,
    relinearizations: int,
    oracle_mode: str,
    upstream_root: Path,
) -> dict[str, Any]:
    for path in (pairs_raw, pairs_sidecar, targets_path):
        if not path.is_file():
            raise FileNotFoundError(f"stale pose rehearsal custody unavailable: {path}")
    sidecar = json.loads(pairs_sidecar.read_text())
    expected_bytes = int(np.prod((CHUNK_PAIRS, *PAIR_SHAPE)))
    if sidecar.get("bytes") != expected_bytes or pairs_raw.stat().st_size != expected_bytes:
        raise RuntimeError("stale composed-frame byte geometry differs")
    if sidecar.get("pair_start") != 0 or sidecar.get("pair_stop") != CHUNK_PAIRS:
        raise RuntimeError("stale composed-frame pair interval differs")
    target_bundle = json.loads(targets_path.read_text())
    if (
        target_bundle.get("pair_count") != 600
        or target_bundle.get("output_dimension") != 6
        or target_bundle.get("rows", [{}])[0].get("pair_id") != 0
    ):
        raise RuntimeError("stale pose target custody differs")
    target = np.asarray(target_bundle["rows"][0]["center"], dtype=np.float64)
    if target.shape != (6,) or not np.all(np.isfinite(target)):
        raise RuntimeError("stale pose target is malformed")

    camera = np.memmap(
        pairs_raw,
        mode="r",
        dtype=np.uint8,
        shape=(CHUNK_PAIRS, *PAIR_SHAPE),
    )
    parent = np.array(camera[0], dtype=np.uint8, copy=True)
    del camera
    rendered = _render_basis(SEED, BASIS_SELECTOR, tuple(parent.shape[1:]))
    geometry: list[tuple[float, float]] = []
    for field in rendered:
        mean = float(np.mean(field, dtype=np.float64))
        centered = np.asarray(field, dtype=np.float64) - mean
        rms = float(np.sqrt(np.mean(centered * centered, dtype=np.float64)))
        geometry.append((mean, rms))
    denominator = float(np.prod(parent.shape[1:]))
    baseline_offset = np.array([4.0, 3.0, 2.0, 1.5, 1.0, 0.5], dtype=np.float64)

    def artifact_for(codes: np.ndarray) -> TerminalPoseCandidateArtifact:
        packet = serialize_terminal_pose_packet(
            TerminalPosePacketV1(
                seed=SEED,
                basis_selector=BASIS_SELECTOR,
                amplitude_q8=AMPLITUDE_Q8,
                coefficients=np.asarray(codes, dtype=np.int16)[None, :],
            )
        )
        return TerminalPoseCandidateArtifact(
            outer_archive=packet,
            terminal_packet=packet,
            scope=CandidateArtifactScope.TERMINAL_SECTION_ONLY,
        )

    frame1_checks: list[bool] = []

    if oracle_mode == "test":

        def scorer(pair: np.ndarray, artifact: TerminalPoseCandidateArtifact) -> PoseJointEvaluation:
            frame1_checks.append(bool(np.array_equal(pair[1], parent[1])))
            delta = pair[0].astype(np.float64) - parent[0].astype(np.float64)
            response = np.empty(6, dtype=np.float64)
            for index, field in enumerate(rendered):
                mean, rms = geometry[index]
                normalized = (np.asarray(field, dtype=np.float64) - mean) / rms
                response[index] = float(np.sum(delta * normalized, dtype=np.float64)) / denominator
            pose = target + baseline_offset + response
            d_pose = float(np.mean((pose - target) ** 2, dtype=np.float64))
            return PoseJointEvaluation(
                pose6=pose,
                d_seg=0.0,
                d_pose=d_pose,
                archive_bytes=artifact.archive_bytes,
                archive_sha256=artifact.archive_sha256,
                sample_count=1,
                authority_marker=STALE_POSE_REHEARSAL_AUTHORITY_MARKER,
                custody_digest=None,
                realized=True,
            )

        oracle_payload = {
            "mode": "deterministic_test_oracle",
            "pose_model_sha256": None,
        }
        evidence_axis = "[deterministic stale mechanism rehearsal]"
        authority_blocker = (
            "The exact stale composed frame and banked six-value target were "
            "used, but a deterministic test oracle supplied pose verdicts. "
            "No frozen scorer, full archive, or public evaluator was invoked."
        )
    elif oracle_mode == "frozen-posenet":
        posenet, posenet_path = _load_frozen_posenet(upstream_root)

        def scorer(pair: np.ndarray, artifact: TerminalPoseCandidateArtifact) -> PoseJointEvaluation:
            frame1_checks.append(bool(np.array_equal(pair[1], parent[1])))
            pose = _frozen_posenet6(posenet, pair)
            d_pose = float(np.mean((pose - target) ** 2, dtype=np.float64))
            return PoseJointEvaluation(
                pose6=pose,
                d_seg=0.0,
                d_pose=d_pose,
                archive_bytes=artifact.archive_bytes,
                archive_sha256=artifact.archive_sha256,
                sample_count=1,
                authority_marker=STALE_POSE_REHEARSAL_AUTHORITY_MARKER,
                custody_digest=None,
                realized=True,
            )

        oracle_payload = {
            "mode": "pinned_upstream_frozen_posenet_cpu",
            "modules_path": str(upstream_root / "modules.py"),
            "modules_sha256": _file_sha256(upstream_root / "modules.py"),
            "pose_model_path": str(posenet_path),
            "pose_model_sha256": _file_sha256(posenet_path),
        }
        evidence_axis = "[macOS-CPU frozen-PoseNet one-pair advisory]"
        authority_blocker = (
            "Pinned frozen PoseNet was evaluated on exactly one stale composed "
            "pair. d_seg=0 encodes the exact frame1-frozen delta but omits the "
            "constant parent d_seg; archive_bytes prices only the terminal "
            "section because candidate artifact equals terminal packet; no full "
            "archive, n600 scorer pass, or public evaluator was invoked."
        )
    else:
        raise ValueError("oracle_mode must be test or frozen-posenet")

    result = solve_terminal_pose_gn(
        parent,
        target,
        _render_basis,
        artifact_for,
        scorer,
        seed=SEED,
        basis_selector=BASIS_SELECTOR,
        config=TerminalPoseGNConfig(
            relinearizations=relinearizations,
            damping=1.0e-3,
            amplitude_q8=AMPLITUDE_Q8,
            authority_mode=PoseAuthorityMode.STALE_REHEARSAL,
        ),
    )
    if not frame1_checks or not all(frame1_checks):
        raise AssertionError("terminal pose rehearsal changed frozen frame 1")
    final_artifact = artifact_for(result.final_coefficients)
    final_packet = final_artifact.terminal_packet
    parsed = parse_terminal_pose_packet(final_packet)
    if not np.array_equal(parsed.coefficients[0], result.final_coefficients):
        raise AssertionError("terminal pose rehearsal final packet differs")

    payload = result.to_payload()
    payload.update(
        {
            "schema": "ddm_eg1_terminal_pose_gn_stale_rehearsal.v1",
            "evidence_axis": evidence_axis,
            "oracle": oracle_payload,
            "promotion_eligible": False,
            "score_claim": False,
            "pointer_moved": False,
            "max_pairs": 1,
            "frame1_frozen_all_realized_verdicts": True,
            "final_packet_bytes": len(final_packet),
            "final_packet_sha256": sha256(final_packet).hexdigest(),
            "final_candidate_artifact": final_artifact.to_payload(),
            "candidate_artifact_scope": "terminal-section-only; outer artifact equals packet",
            "stale_substrate": {
                "pairs_raw": str(pairs_raw),
                "pairs_sidecar": str(pairs_sidecar),
                "pairs_raw_bytes": pairs_raw.stat().st_size,
                "pairs_raw_sha256_from_sidecar": sidecar["sha256"],
                "pairs_sidecar_sha256": _file_sha256(pairs_sidecar),
                "state_sha256": sidecar["state_sha256"],
                "pair_id": 0,
                "pair_shape": list(PAIR_SHAPE),
                "target_bundle": str(targets_path),
                "target_bundle_sha256": _file_sha256(targets_path),
                "target_pose6": target.tolist(),
            },
            "basis_provenance": {
                "kind": "generic analytic lowest-frequency cosine fields",
                "rank": 6,
                "seed": SEED,
                "basis_selector": BASIS_SELECTOR,
                "amplitude_q8": AMPLITUDE_Q8,
                "pr130_transfer": (
                    "lessons only: low-frequency carrier, per-field "
                    "zero-mean/unit-RMS normalization, per-pair coefficients"
                ),
                "copied_pr130_constants_or_payload": False,
            },
            "authority_blocker": authority_blocker,
        }
    )
    return payload


def rehearse_both(
    pairs_raw: Path,
    pairs_sidecar: Path,
    targets_path: Path,
    *,
    relinearizations: int,
    upstream_root: Path,
) -> dict[str, Any]:
    """Produce the complete v2 receipt in one deterministic invocation."""

    mechanism = rehearse(
        pairs_raw,
        pairs_sidecar,
        targets_path,
        relinearizations=relinearizations,
        oracle_mode="test",
        upstream_root=upstream_root,
    )
    frozen = rehearse(
        pairs_raw,
        pairs_sidecar,
        targets_path,
        relinearizations=relinearizations,
        oracle_mode="frozen-posenet",
        upstream_root=upstream_root,
    )
    shared_keys = ("stale_substrate", "basis_provenance")
    for key in shared_keys:
        if mechanism[key] != frozen[key]:
            raise AssertionError(f"terminal pose shared receipt field differs: {key}")

    def nested(payload: dict[str, Any], *, oracle: str) -> dict[str, Any]:
        result = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "schema",
                "stale_substrate",
                "basis_provenance",
                "authority_blocker",
                "promotion_eligible",
                "pointer_moved",
            }
        }
        result["rehearsal_command"] = (
            "PYTHONPATH=src uv run --no-project python "
            "tools/rehearse_terminal_pose_gn.py --max-pairs 1 "
            f"--relinearizations {relinearizations} --oracle {oracle}"
        )
        return result

    return {
        "schema": "ddm_eg1_terminal_pose_gn_stale_rehearsal.v2",
        "generated_by": "tools/rehearse_terminal_pose_gn.py --oracle both",
        "promotion_eligible": False,
        "production_accepted": False,
        "promotion_allowed": False,
        "score_claim": False,
        "pointer_moved": False,
        "max_pairs": 1,
        "stale_substrate": mechanism["stale_substrate"],
        "basis_provenance": {
            **mechanism["basis_provenance"],
            "basis_key_contract": ["seed", "basis_selector", "frame_shape"],
        },
        "mechanism_canary": nested(mechanism, oracle="test"),
        "frozen_posenet_rehearsal": nested(frozen, oracle="frozen-posenet"),
        "authority_blocker": frozen["authority_blocker"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-raw", type=Path, default=DEFAULT_PAIRS_RAW)
    parser.add_argument("--pairs-sidecar", type=Path, default=DEFAULT_PAIRS_SIDECAR)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM_ROOT)
    parser.add_argument("--max-pairs", type=int, choices=(1,), default=1)
    parser.add_argument("--relinearizations", type=int, choices=(2, 3), default=2)
    parser.add_argument("--oracle", choices=("test", "frozen-posenet", "both"), default="both")
    args = parser.parse_args()
    if args.oracle == "both":
        payload = rehearse_both(
            args.pairs_raw,
            args.pairs_sidecar,
            args.targets,
            relinearizations=args.relinearizations,
            upstream_root=args.upstream_root,
        )
    else:
        payload = rehearse(
            args.pairs_raw,
            args.pairs_sidecar,
            args.targets,
            relinearizations=args.relinearizations,
            oracle_mode=args.oracle,
            upstream_root=args.upstream_root,
        )
    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
