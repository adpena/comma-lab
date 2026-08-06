# SPDX-License-Identifier: MIT
"""ddm_tq1 optimal-form token-edit Phase A builder.

This arm is intentionally scorer-light until the jd5 endpoint receipt opens the
Phase B scorer slot.  It builds a derived per-cell token-edit menu, computes the
true affected pair set on the receiver token lattice, prices candidates through
the real IX2 payload/ZIP path, and writes the fire order for the realized
acceptance pass.  It does not claim a score from cached instruments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
EXP = REPO / "experiments"
for path in (str(SRC), str(EXP)):
    if path not in sys.path:
        sys.path.insert(0, path)

from tac.optimization import ddm_ix2_archive_container as IX2  # noqa: E402
from tac.submission_chain import build_byte_ledger, sha256_file  # noqa: E402

import ddm_td1_token_drop_guided_surface as TD1  # noqa: E402


DEN = 37_545_489
AXIS = "[macOS-CPU frozen-scorer advisory]"
BASE_ARCHIVE_SHA256 = "d5e814d5b9f65c3094b0e65fecdd7771734d03c420c63d1d2033a671b766986a"
BASELINE = {
    "source": "main_hot_state live own-vehicle row / qo1 sub_auto_pairbit",
    "archive_bytes": 357_836,
    "archive_sha256": BASE_ARCHIVE_SHA256,
    "d_seg": 0.00431179,
    "d_pose": 0.00071459,
    "score": 0.7539807296911207,
    "axis": AXIS,
    "score_claim": False,
}
DOMINATED_GLOBAL_BASELINE = (16, 12, 8, 4)
R8_POSE_TERM_EROSION_LIMIT = 0.005
CHECKPOINT_EVERY = 10


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    row: int
    col: int
    direction: str
    rung: int
    priority: float
    joint_guard: float
    seg_guard: float
    pose_guard: float
    activity: float
    affected_pair_count: int
    affected_pair_preview: tuple[int, ...]


@dataclass(frozen=True)
class ComponentScore:
    d_seg: float
    d_pose: float
    archive_bytes: int
    axis: str = AXIS
    score_claim: bool = False

    @property
    def score(self) -> float:
        return score_from_components(self.d_seg, self.d_pose, self.archive_bytes)

    @property
    def pose_term(self) -> float:
        return math.sqrt(10.0 * self.d_pose)


def score_from_components(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return 100.0 * float(d_seg) + math.sqrt(10.0 * float(d_pose)) + 25.0 * int(archive_bytes) / DEN


def acceptance_verdict(
    current: ComponentScore,
    candidate: ComponentScore,
    *,
    pose_term_erosion_limit: float = R8_POSE_TERM_EROSION_LIMIT,
) -> dict[str, Any]:
    """Accept only a realized scorer row whose joint score strictly decreases."""

    delta_s = candidate.score - current.score
    pose_term_erosion = candidate.pose_term - current.pose_term
    accepted = bool(delta_s < 0.0 and pose_term_erosion <= pose_term_erosion_limit)
    return {
        "schema": "ddm_tq1_realized_acceptance_verdict.v1",
        "accepted": accepted,
        "delta_S": delta_s,
        "delta_d_seg": float(candidate.d_seg - current.d_seg),
        "delta_d_pose": float(candidate.d_pose - current.d_pose),
        "delta_archive_bytes": int(candidate.archive_bytes - current.archive_bytes),
        "current_score": current.score,
        "candidate_score": candidate.score,
        "pose_term_erosion": pose_term_erosion,
        "pose_term_erosion_limit": pose_term_erosion_limit,
        "axis": candidate.axis,
        "score_claim": bool(candidate.score_claim),
        "authority": "requires real receiver -> R -> uint8 -> frozen scorer components",
    }


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "__dict__"):
        return asdict(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot serialize {type(obj)!r}")


def write_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, default=_json_default, allow_nan=False) + "\n")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=_json_default, allow_nan=False) + "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    value = np.ascontiguousarray(arr)
    return hashlib.sha256(value.tobytes()).hexdigest()


def read_payload_archive(archive: Path) -> tuple[bytes, list[bytes]]:
    with zipfile.ZipFile(archive) as zf:
        payload = zf.read("0.bin")
    bulk, sections = IX2.parse_payload(payload)
    return bulk, list(sections)


def codes_to_pm1(codes16: np.ndarray) -> np.ndarray:
    return np.asarray(codes16, dtype=np.float64) / 15.0 * 2.0 - 1.0


def pm1_to_codes16(values: np.ndarray) -> np.ndarray:
    x01 = (np.clip(values, -1.0, 1.0) + 1.0) * 0.5
    return np.rint(x01 * 15.0).astype(np.uint8)


def snap_codes_to_sublattice(codes16: np.ndarray, levels: int) -> np.ndarray:
    if not 2 <= int(levels) <= 16:
        raise ValueError(f"sublattice level must be in [2, 16], got {levels}")
    t = np.clip(codes_to_pm1(codes16), -1.0, 1.0)
    denom = float(int(levels) - 1)
    x01 = (t + 1.0) * 0.5
    snapped = np.round(x01 * denom) / denom * 2.0 - 1.0
    return pm1_to_codes16(snapped)


def mode_step_codes(cell_codes: np.ndarray, mode: np.ndarray, *, max_step: int) -> np.ndarray:
    """Move each token one or more circular-lattice steps toward its cell mode."""

    if max_step < 1:
        raise ValueError("max_step must be positive")
    values = np.asarray(cell_codes, dtype=np.uint8)
    target = np.asarray(mode, dtype=np.uint8)[None, :]
    forward = (target.astype(np.int16) - values.astype(np.int16)) % 16
    backward = (values.astype(np.int16) - target.astype(np.int16)) % 16
    take_forward = forward <= backward
    delta = np.where(take_forward, np.minimum(forward, max_step), -np.minimum(backward, max_step))
    return ((values.astype(np.int16) + delta) % 16).astype(np.uint8)


def affected_pairs_for_cell(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Return exact pair ids whose receiver token cell changed."""

    lhs = np.asarray(before, dtype=np.uint8)
    rhs = np.asarray(after, dtype=np.uint8)
    if lhs.shape != rhs.shape or lhs.ndim != 2:
        raise ValueError(f"expected matching (pairs, channels) cells, got {lhs.shape} and {rhs.shape}")
    return np.flatnonzero(np.any(lhs != rhs, axis=1)).astype(np.int64)


def normalized_rank(field: np.ndarray) -> np.ndarray:
    flat = np.asarray(field, dtype=np.float64).reshape(-1)
    if flat.size <= 1:
        return np.zeros_like(np.asarray(field, dtype=np.float64))
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(flat, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, flat.size)
    return ranks.reshape(np.asarray(field).shape)


def derive_cell_rungs(
    *,
    joint_guard: np.ndarray,
    seg_guard: np.ndarray,
    pose_guard: np.ndarray,
    activity: np.ndarray,
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Per-cell adaptive rungs from measured custody fields, not a global menu."""

    risk = normalized_rank(joint_guard)
    act = normalized_rank(activity)
    seg = normalized_rank(seg_guard)
    pose = normalized_rank(pose_guard)
    out: dict[tuple[int, int], tuple[int, ...]] = {}
    for r in range(joint_guard.shape[0]):
        for c in range(joint_guard.shape[1]):
            safety = 1.0 - float(risk[r, c])
            byte_need = 0.35 + 0.65 * float(act[r, c])
            depth = safety * byte_need
            if depth < 0.08:
                levels: list[int] = []
            elif depth < 0.18:
                levels = [15]
            elif depth < 0.32:
                levels = [15, 14]
            elif depth < 0.50:
                levels = [15, 14, 13, 12]
            elif depth < 0.70:
                levels = [14, 13, 12, 10, 8]
            else:
                levels = [13, 12, 10, 8, 6, 4]

            if float(seg[r, c]) > 0.88:
                levels = [lvl for lvl in levels if lvl >= 14]
            elif float(seg[r, c]) > 0.74:
                levels = [lvl for lvl in levels if lvl >= 12]
            if float(pose[r, c]) > 0.86:
                levels = [lvl for lvl in levels if lvl >= 12]
            elif float(pose[r, c]) > 0.72:
                levels = [lvl for lvl in levels if lvl >= 10]

            # Remove duplicates without sorting away the derived depth order.
            deduped: list[int] = []
            for level in levels:
                if level not in deduped:
                    deduped.append(level)
            out[(r, c)] = tuple(deduped)
    return out


def mutate_cell(tokens: np.ndarray, row: int, col: int, direction: str, rung: int, mode: np.ndarray) -> np.ndarray:
    before = np.asarray(tokens[:, row, col, :], dtype=np.uint8)
    if direction == "snap_sublattice":
        return snap_codes_to_sublattice(before, rung)
    if direction == "mode_step":
        return mode_step_codes(before, mode[row, col, :], max_step=rung)
    if direction == "mode_drop":
        return np.broadcast_to(mode[row, col, :], before.shape).copy().astype(np.uint8)
    raise ValueError(f"unknown direction {direction!r}")


def apply_candidate(tokens: np.ndarray, candidate: Candidate, mode: np.ndarray) -> np.ndarray:
    mutated = np.array(tokens, copy=True)
    mutated[:, candidate.row, candidate.col, :] = mutate_cell(
        tokens, candidate.row, candidate.col, candidate.direction, candidate.rung, mode
    )
    return np.ascontiguousarray(mutated)


def build_candidates(
    *,
    tokens: np.ndarray,
    fields: Mapping[str, np.ndarray],
) -> tuple[list[Candidate], np.ndarray, dict[tuple[int, int], tuple[int, ...]]]:
    mode, _delta = IX2._factor_mode_delta(tokens, 16)
    rungs = derive_cell_rungs(
        joint_guard=np.asarray(fields["joint_guard"]),
        seg_guard=np.asarray(fields["seg_guard"]),
        pose_guard=np.asarray(fields["pose_guard"]),
        activity=np.asarray(fields["activity"]),
    )
    joint_rank = normalized_rank(np.asarray(fields["joint_guard"]))
    activity_rank = normalized_rank(np.asarray(fields["activity"]))
    pose_rank = normalized_rank(np.asarray(fields["pose_guard"]))

    rows: list[Candidate] = []
    for r in range(tokens.shape[1]):
        for c in range(tokens.shape[2]):
            cell_levels = rungs[(r, c)]
            for level in cell_levels:
                after = mutate_cell(tokens, r, c, "snap_sublattice", level, mode)
                affected = affected_pairs_for_cell(tokens[:, r, c, :], after)
                if affected.size == 0:
                    continue
                depth = (16.0 - level) / 12.0
                priority = depth * (0.2 + float(activity_rank[r, c])) / (0.05 + float(joint_rank[r, c]))
                rows.append(
                    Candidate(
                        candidate_id=f"snap_r{r:02d}_c{c:02d}_L{level:02d}",
                        row=r,
                        col=c,
                        direction="snap_sublattice",
                        rung=int(level),
                        priority=float(priority),
                        joint_guard=float(fields["joint_guard"][r, c]),
                        seg_guard=float(fields["seg_guard"][r, c]),
                        pose_guard=float(fields["pose_guard"][r, c]),
                        activity=float(fields["activity"][r, c]),
                        affected_pair_count=int(affected.size),
                        affected_pair_preview=tuple(int(x) for x in affected[:16]),
                    )
                )

            if float(joint_rank[r, c]) < 0.45 and float(activity_rank[r, c]) > 0.30:
                for step in (1, 2):
                    after = mutate_cell(tokens, r, c, "mode_step", step, mode)
                    affected = affected_pairs_for_cell(tokens[:, r, c, :], after)
                    if affected.size == 0:
                        continue
                    priority = (0.8 + 0.3 * step) * (0.25 + float(activity_rank[r, c])) / (
                        0.07 + float(joint_rank[r, c])
                    )
                    rows.append(
                        Candidate(
                            candidate_id=f"step{step}_r{r:02d}_c{c:02d}",
                            row=r,
                            col=c,
                            direction="mode_step",
                            rung=step,
                            priority=float(priority),
                            joint_guard=float(fields["joint_guard"][r, c]),
                            seg_guard=float(fields["seg_guard"][r, c]),
                            pose_guard=float(fields["pose_guard"][r, c]),
                            activity=float(fields["activity"][r, c]),
                            affected_pair_count=int(affected.size),
                            affected_pair_preview=tuple(int(x) for x in affected[:16]),
                        )
                    )

            if float(joint_rank[r, c]) < 0.16 and float(pose_rank[r, c]) < 0.35 and float(activity_rank[r, c]) > 0.50:
                after = mutate_cell(tokens, r, c, "mode_drop", 0, mode)
                affected = affected_pairs_for_cell(tokens[:, r, c, :], after)
                if affected.size:
                    rows.append(
                        Candidate(
                            candidate_id=f"mode_r{r:02d}_c{c:02d}",
                            row=r,
                            col=c,
                            direction="mode_drop",
                            rung=0,
                            priority=float((1.6 + float(activity_rank[r, c])) / (0.08 + float(joint_rank[r, c]))),
                            joint_guard=float(fields["joint_guard"][r, c]),
                            seg_guard=float(fields["seg_guard"][r, c]),
                            pose_guard=float(fields["pose_guard"][r, c]),
                            activity=float(fields["activity"][r, c]),
                            affected_pair_count=int(affected.size),
                            affected_pair_preview=tuple(int(x) for x in affected[:16]),
                        )
                    )

    rows.sort(key=lambda cand: (-cand.priority, cand.joint_guard, cand.affected_pair_count, cand.candidate_id))
    return rows, mode, rungs


def write_candidate_menu(out_root: Path, candidates: Sequence[Candidate]) -> Path:
    menu_path = out_root / "tq1_phase_a_candidate_menu.jsonl"
    if menu_path.exists():
        menu_path.unlink()
    for idx, candidate in enumerate(candidates, start=1):
        append_jsonl(
            menu_path,
            {
                "schema": "ddm_tq1_phase_a_candidate_menu.v1",
                "index": idx,
                "candidate": asdict(candidate),
                "axis": AXIS,
                "score_claim": False,
                "candidate_source": "receiver token lattice decoded from qo1 archive",
            },
        )
    return menu_path


def build_archive_bytes(tokens: np.ndarray, sections: Sequence[bytes]) -> tuple[bytes, bytes]:
    token_frame = IX2.encode_token_frame(tokens, levels=16)
    payload = IX2.build_payload(token_frame, list(sections))
    return token_frame, IX2.build_single_member_zip(payload)


def write_smoke_archive(root: Path, candidate_id: str, archive_bytes: bytes) -> dict[str, Any]:
    candidate_dir = root / "smoke_candidates" / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    archive_path = candidate_dir / "archive.zip"
    archive_path.write_bytes(archive_bytes)
    ledger = build_byte_ledger(archive_path)
    return {
        "candidate_dir": str(candidate_dir),
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "byte_ledger": {
            "closes": ledger.closes(),
            "residual_bytes": ledger.residual_bytes,
            "payload_reencodes_identically": ledger.payload_reencodes_identically,
            "bulk_bytes": ledger.bulk_bytes,
            "joint_coded_bytes": ledger.joint_coded_bytes,
            "archive_bytes": ledger.archive_bytes,
            "archive_sha256": ledger.archive_sha256,
        },
    }


def price_candidates(
    *,
    tokens: np.ndarray,
    sections: Sequence[bytes],
    candidates: Sequence[Candidate],
    mode: np.ndarray,
    out_root: Path,
    base_archive_bytes: int,
    smoke_moves: int,
) -> tuple[list[dict[str, Any]], Path]:
    ledger_path = out_root / "tq1_phase_a_candidate_prices.jsonl"
    if ledger_path.exists():
        ledger_path.unlink()
    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        mutated = apply_candidate(tokens, candidate, mode)
        token_frame, archive_bytes = build_archive_bytes(mutated, sections)
        candidate_row: dict[str, Any] = {
            "schema": "ddm_tq1_phase_a_candidate_price.v1",
            "index": idx,
            "candidate": asdict(candidate),
            "candidate_tokens_sha256": sha256_array(mutated),
            "token_frame_bytes": len(token_frame),
            "token_frame_sha256": sha256_bytes(token_frame),
            "archive_bytes": len(archive_bytes),
            "archive_sha256": sha256_bytes(archive_bytes),
            "delta_archive_bytes_vs_qo1": int(len(archive_bytes) - base_archive_bytes),
            "rate_delta_S_measured_bytes": 25.0 * (len(archive_bytes) - base_archive_bytes) / DEN,
            "accepted": False,
            "acceptance_status": "NOT_RUN_PHASE_A_SCORER_LIGHT",
            "realized_components": None,
            "axis": AXIS,
            "score_claim": False,
        }
        if idx <= smoke_moves:
            candidate_row["smoke_byteclose"] = write_smoke_archive(out_root, candidate.candidate_id, archive_bytes)
        append_jsonl(ledger_path, candidate_row)
        rows.append(candidate_row)
        if idx % CHECKPOINT_EVERY == 0 or idx == len(candidates):
            write_json(
                out_root / "checkpoints" / f"phase_a_price_{idx:04d}.json",
                {
                    "schema": "ddm_tq1_phase_a_checkpoint.v1",
                    "checkpoint_index": idx,
                    "total_planned_candidates": len(candidates),
                    "ledger_path": str(ledger_path),
                    "last_candidate_id": candidate.candidate_id,
                    "score_claim": False,
                    "axis": AXIS,
                },
            )
    return rows, ledger_path


def discover_jd5_endpoint(jd5_dir: Path) -> dict[str, Any]:
    paths = sorted(jd5_dir.glob("*jd5*endpoint*.json")) + sorted(jd5_dir.glob("*endpoint*jd5*.json"))
    seen: set[Path] = set()
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            rows.append({"path": str(path), "status": "UNREADABLE", "error": repr(exc)})
            continue
        rows.append({"path": str(path), "status": data.get("status"), "sha256": sha256_file(path)})
        if data.get("status") == "complete":
            return {"ready": True, "complete_endpoint": rows[-1], "checked": rows}
    return {"ready": False, "complete_endpoint": None, "checked": rows}


def phase_b_psutil_preflight(limit_gb: float = 20.0) -> dict[str, Any]:
    import psutil

    rss = psutil.Process(os.getpid()).memory_info().rss
    limit = int(limit_gb * (1024**3))
    return {
        "rss_bytes": int(rss),
        "rss_limit_bytes": limit,
        "rss_limit_gb": limit_gb,
        "passes": bool(rss <= limit),
    }


def apply_realized_measurement_jsonl(path: Path) -> dict[str, Any]:
    current = ComponentScore(
        d_seg=float(BASELINE["d_seg"]),
        d_pose=float(BASELINE["d_pose"]),
        archive_bytes=int(BASELINE["archive_bytes"]),
        axis=AXIS,
        score_claim=False,
    )
    accepted = 0
    total = 0
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            raw = json.loads(line)
            candidate = ComponentScore(
                d_seg=float(raw["d_seg"]),
                d_pose=float(raw["d_pose"]),
                archive_bytes=int(raw["archive_bytes"]),
                axis=str(raw.get("axis", AXIS)),
                score_claim=bool(raw.get("score_claim", False)),
            )
            verdict = acceptance_verdict(current, candidate)
            if verdict["accepted"]:
                current = candidate
                accepted += 1
            rows.append({"input": raw, "verdict": verdict, "post_current_score": current.score})
    return {
        "schema": "ddm_tq1_phase_b_realized_acceptance_from_measurement_jsonl.v1",
        "measurement_jsonl": str(path),
        "rows": total,
        "accepted": accepted,
        "final_current": asdict(current) | {"score": current.score},
        "verdict_rows": rows,
        "score_claim": False,
    }


def _section_sha256(sections: Sequence[bytes]) -> list[str]:
    return [sha256_bytes(section) for section in sections]


def load_phase_a_inputs(args: argparse.Namespace) -> tuple[bytes, list[bytes], np.ndarray, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    base_archive = args.base_sub / "archive.zip"
    if not base_archive.exists():
        raise FileNotFoundError(f"base archive missing: {base_archive}")
    got_sha = sha256_file(base_archive)
    if got_sha != BASE_ARCHIVE_SHA256:
        raise RuntimeError(f"refusing cross-object TQ1: base sha {got_sha} != {BASE_ARCHIVE_SHA256}")

    bulk, sections = read_payload_archive(base_archive)
    tokens = IX2.decode_token_frame(bulk)
    if tokens.shape != (600, 24, 32, 4):
        raise ValueError(f"live token shape drift: {tokens.shape}")

    token_source_equal = None
    token_source_sha = None
    if args.token_source.exists():
        source_tokens = np.load(args.token_source)
        token_source_equal = bool(np.array_equal(tokens, source_tokens))
        token_source_sha = sha256_file(args.token_source)
        if not token_source_equal:
            raise RuntimeError("live qo1 token bulk differs from token source; refusing mixed-object menu")

    instrument_source, joint_guard, field_hashes, fields = TD1._instrument_fields(
        tokens=tokens,
        gt_argmax=args.gt_argmax,
        cx1_argmax=args.cx1_argmax,
        g3_jsonl=args.g3_jsonl,
        g4_recurrence=args.g4_recurrence,
        sg1_cell_flip_mass=args.sg1_cell_flip_mass,
    )
    fields = dict(fields)
    fields["joint_guard"] = joint_guard

    base_payload = IX2.build_payload(IX2.encode_token_frame(tokens, levels=16), sections)
    restaged_archive = IX2.build_single_member_zip(base_payload)
    restaged_sha = sha256_bytes(restaged_archive)
    if restaged_sha != got_sha:
        raise RuntimeError(
            "base restage did not reproduce qo1 archive; refusing candidate deltas "
            f"(restaged {restaged_sha}, live {got_sha})"
        )

    base_info = {
        "path": str(base_archive),
        "sha256": got_sha,
        "bytes": base_archive.stat().st_size,
        "bulk_bytes": len(bulk),
        "bulk_sha256": sha256_bytes(bulk),
        "tokens_sha256": sha256_array(tokens),
        "token_source": {
            "path": str(args.token_source),
            "exists": args.token_source.exists(),
            "sha256": token_source_sha,
            "equal_to_archive_tokens": token_source_equal,
        },
        "restaged_archive_sha256": restaged_sha,
        "restaged_archive_bytes": len(restaged_archive),
        "restaged_matches_live": True,
        "section_bytes": [len(section) for section in sections],
        "section_sha256": _section_sha256(sections),
    }
    return bulk, sections, tokens, fields, {"instrument_source": instrument_source, "base_info": base_info}, field_hashes


def rung_summary(rungs: Mapping[tuple[int, int], tuple[int, ...]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    unique_menus: dict[str, int] = {}
    for menu in rungs.values():
        unique_menus[",".join(str(x) for x in menu) if menu else "preserve"] = (
            unique_menus.get(",".join(str(x) for x in menu) if menu else "preserve", 0) + 1
        )
        for level in menu:
            counts[str(level)] = counts.get(str(level), 0) + 1
    return {
        "cells": len(rungs),
        "rung_use_counts": counts,
        "unique_per_cell_menus": unique_menus,
        "global_16_12_8_4_used_as_generator": False,
        "dominated_global_baseline_recorded_only": list(DOMINATED_GLOBAL_BASELINE),
    }


def write_receipts(
    *,
    args: argparse.Namespace,
    out_root: Path,
    receipt_dir: Path,
    base_info: Mapping[str, Any],
    instrument_source: Mapping[str, Any],
    field_hashes: Sequence[Mapping[str, Any]],
    candidates: Sequence[Candidate],
    priced_rows: Sequence[Mapping[str, Any]],
    menu_path: Path,
    price_ledger: Path,
    rungs: Mapping[tuple[int, int], tuple[int, ...]],
    jd5_gate: Mapping[str, Any],
    seconds: float,
) -> tuple[Path, Path, Path]:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    followon_status = "QUEUED-WITH-FIRE-ORDER"
    phase_a_json = {
        "schema": "ddm_tq1_phase_a_receipt.v1",
        "score_claim": False,
        "promotion_eligible": False,
        "axis": AXIS,
        "seconds": seconds,
        "phase": "A_BUILD_SCORER_LIGHT",
        "phase_b_status": "GATED_CLOSED_JD5_ENDPOINT_ABSENT" if not jd5_gate["ready"] else "GATED_OPEN_NOT_FIRED_BY_PHASE_A",
        "followon_status": followon_status,
        "baseline": BASELINE,
        "base_archive": base_info,
        "instrument_sources": instrument_source,
        "field_hashes": list(field_hashes),
        "rung_summary": rung_summary(rungs),
        "candidate_menu": {
            "generated": len(candidates),
            "menu_ledger": str(menu_path),
            "priced": len(priced_rows),
            "price_ledger": str(price_ledger),
            "top_candidate_ids": [row["candidate"]["candidate_id"] for row in priced_rows[:10]],
            "accepted_count": 0,
            "accepted_status": "NOT_RUN_PHASE_A_SCORER_LIGHT",
        },
        "jd5_gate": jd5_gate,
        "build_sites": {
            "driver": "experiments/ddm_tq1_optimal_token_edit.py",
            "ssd_root": str(out_root),
            "price_ledger": str(price_ledger),
            "candidate_menu": str(menu_path),
            "smoke_archives": str(out_root / "smoke_candidates"),
            "checkpoints": str(out_root / "checkpoints"),
        },
        "harness_reuse_evidence": [
            "Reuses TD1 _instrument_fields for gt/cx1 argmax, g3 score atlas, g4 recurrence, sg1 flip mass custody.",
            "Reuses live IX2 parse/decode/encode/build_payload/build_single_member_zip receiver/archive path.",
            "Uses #869 cell-drop waterfill invariant: per-candidate bytes are repriced from the current token object, not inferred from singleton tables.",
            "Phase B fire order binds to v19/v19b realized move-level acceptance before any family closure.",
        ],
        "not_measured": {
            "subset_or_n600_scorer": "not run; jd5 endpoint receipt with status=complete was absent",
            "family_closure": "not claimed; zero accepted moves can close only after full derived menu is scorer-realized",
        },
    }
    phase_a_json_path = receipt_dir / "phase_a_receipt.json"
    write_json(phase_a_json_path, phase_a_json)

    next_path = receipt_dir / "NEXT_IF_RESUMED.md"
    next_path.write_text(
        "\n".join(
            [
                "# ddm_tq1 NEXT_IF_RESUMED",
                "",
                "Status: QUEUED-WITH-FIRE-ORDER. Phase A built the derived menu and byte-priced smoke candidates; Phase B is gated on a jd5 endpoint receipt with `status=complete`.",
                "",
                "1. Re-check the gate without touching the jd5 run dir:",
                "   `.venv/bin/python experiments/ddm_tq1_optimal_token_edit.py --phase-b --require-jd5`",
                "2. If the gate opens, wire/run the v19/v19b realized move-level scorer stack on the queued candidate ledger:",
                f"   `{price_ledger}`",
                "3. Accept a move only when its realized receiver -> R -> uint8 -> frozen-scorer components give joint `delta_S < 0` and pose-term erosion <= 0.005.",
                "4. Checkpoint about every 20 accepted/rejected scorer moves under the SSD root before any n600 greedy-to-saturation run.",
                "5. If a full derived menu produces zero accepted realized moves, close the family at optimal-form scope; otherwise stage the accepted endpoint for n600.",
                "",
                "No score is claimed by this Phase A receipt.",
                "",
            ]
        )
    )

    recall_lines = [
        "# ddm_tq1 RECEIPT",
        "",
        "score_claim: false",
        f"axis: {AXIS}",
        f"phase: A_BUILD_SCORER_LIGHT ({seconds:.3f}s)",
        f"phase_b: {'GATED_OPEN_NOT_FIRED_BY_PHASE_A' if jd5_gate['ready'] else 'GATED_CLOSED_JD5_ENDPOINT_ABSENT'}",
        f"follow_on: {followon_status}",
        "",
        "## BUILD SITES",
        "",
        f"- Driver: `experiments/ddm_tq1_optimal_token_edit.py`",
        f"- SSD root: `{out_root}`",
        f"- Price ledger: `{price_ledger}`",
        f"- Candidate menu: `{menu_path}`",
        f"- Phase A JSON: `{phase_a_json_path}`",
        f"- Resume instructions: `{next_path}`",
        "",
        "## PHASE A RESULT",
        "",
        f"- Base archive verified: `{base_info['path']}` sha256 `{base_info['sha256']}`, {base_info['bytes']} B.",
        f"- Candidate menu generated: {len(candidates)} moves with exact affected-pair sets from the decoded receiver token lattice.",
        f"- Candidate prefix priced: {len(priced_rows)} moves; accepted moves: 0 (`NOT_RUN_PHASE_A_SCORER_LIGHT`).",
        f"- Smoke byte-close archives: {min(len(priced_rows), int(args.smoke_moves))}; each parsed with `build_byte_ledger`.",
        "- Global `[16,12,8,4]` is recorded only as a dominated baseline, not used as the generator.",
        "",
        "## PHASE B",
        "",
        "- Status: QUEUED-WITH-FIRE-ORDER.",
        "- Gate checked: no jd5 endpoint receipt with `status=complete` was found." if not jd5_gate["ready"] else "- Gate checked: jd5 endpoint is complete, but Phase A did not spend the scorer slot.",
        "- Required acceptance rule is implemented as `acceptance_verdict`: realized joint `delta_S < 0`, pose-term erosion <= 0.005, real archive bytes.",
        "",
        "## RECALL EVIDENCE",
        "",
        "- `.omx/tmp/codex_runs/tq1_prompt.md`: Phase A build now, Phase B only after jd5 complete endpoint; prior blanket token edits are instance negatives.",
        "- `.omx/tmp/codex_runs/_common_contract.md`: serializer commit, two review passes for `.py`, no forbidden-file edits, no `/tmp` persisted evidence.",
        "- `.omx/state/main_hot_state.md`: qo1 live own-vehicle row and fresh rt1/fz4/ed2 negatives; scorer boundary remains jd5/sq2 ordered.",
        "- `.omx/research/ddm_tq1_preempted_by_rt1_and_sl2_composition_20260805.md`: earlier tq1 preemption is stale for this charter because it only replayed blanket-map negatives.",
        "- `experiments/ddm_td1_token_drop_guided_surface.py`: reused cached scorer-instrument guard fields and IX2 staging pattern.",
        "- `experiments/ddm_tw1_token_waterfill_state_dependence.py`: incorporated the state-dependence lesson by repricing candidates on the actual token object.",
        "- `tools/measure_ddm_dr2b_tolerance_costate.py` and `tools/measure_ddm_rd1_lambda_continuation_frontier.py`: Phase B fire order must reuse their v19/v19b realized move-level accounting before closure.",
        "",
        "## FOLLOW-ONS",
        "",
        "- QUEUED-WITH-FIRE-ORDER: run Phase B after jd5 complete endpoint, with psutil RSS preflight <= 20 GB and checkpoint every about 20 scorer moves.",
        "- FOLDED: previous rt1/fz4/ed2 blanket negatives are dominated-baseline context only; they do not close the TQ1 family.",
        "",
    ]
    receipt_path = receipt_dir / "RECEIPT.md"
    receipt_path.write_text("\n".join(recall_lines))
    return receipt_path, next_path, phase_a_json_path


def run_phase_a(args: argparse.Namespace) -> int:
    started = time.time()
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)

    _bulk, sections, tokens, fields, sources, field_hashes = load_phase_a_inputs(args)
    candidates, mode, rungs = build_candidates(tokens=tokens, fields=fields)
    if not candidates:
        raise RuntimeError("derived menu produced no candidates")
    menu_path = write_candidate_menu(args.out_root, candidates)
    candidates_to_price = candidates[: args.price_top]

    priced_rows, price_ledger = price_candidates(
        tokens=tokens,
        sections=sections,
        candidates=candidates_to_price,
        mode=mode,
        out_root=args.out_root,
        base_archive_bytes=int(sources["base_info"]["bytes"]),
        smoke_moves=args.smoke_moves,
    )
    jd5_gate = discover_jd5_endpoint(args.jd5_dir)
    receipt_path, next_path, phase_a_json_path = write_receipts(
        args=args,
        out_root=args.out_root,
        receipt_dir=args.receipt_dir,
        base_info=sources["base_info"],
        instrument_source=sources["instrument_source"],
        field_hashes=field_hashes,
        candidates=candidates,
        priced_rows=priced_rows,
        menu_path=menu_path,
        price_ledger=price_ledger,
        rungs=rungs,
        jd5_gate=jd5_gate,
        seconds=time.time() - started,
    )
    print(
        json.dumps(
            {
                "schema": "ddm_tq1_phase_a_stdout.v1",
                "score_claim": False,
                "phase": "A_BUILD_SCORER_LIGHT",
                "priced_candidates": len(priced_rows),
                "receipt": str(receipt_path),
                "next_if_resumed": str(next_path),
                "phase_a_json": str(phase_a_json_path),
                "jd5_ready": jd5_gate["ready"],
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


def run_phase_b(args: argparse.Namespace) -> int:
    jd5_gate = discover_jd5_endpoint(args.jd5_dir)
    if args.require_jd5 and not jd5_gate["ready"]:
        args.receipt_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            args.receipt_dir / "phase_b_queued_gate_absent.json",
            {
                "schema": "ddm_tq1_phase_b_gate_check.v1",
                "status": "QUEUED-WITH-FIRE-ORDER",
                "reason": "jd5 endpoint receipt with status=complete not found",
                "jd5_gate": jd5_gate,
                "score_claim": False,
            },
        )
        print(json.dumps({"status": "QUEUED-WITH-FIRE-ORDER", "jd5_ready": False}, indent=1))
        return 0

    preflight = phase_b_psutil_preflight()
    if not preflight["passes"]:
        raise RuntimeError(f"Phase B psutil RSS preflight failed: {preflight}")
    if args.realized_measurement_jsonl is None:
        raise RuntimeError(
            "Phase B gate is open, but no realized scorer measurement JSONL was supplied. "
            "Do not fabricate acceptance; wire the v19/v19b receiver->R->uint8 scorer stack first."
        )
    result = apply_realized_measurement_jsonl(args.realized_measurement_jsonl)
    result["psutil_preflight"] = preflight
    result["jd5_gate"] = jd5_gate
    write_json(args.receipt_dir / "phase_b_realized_acceptance_receipt.json", result)
    print(json.dumps({"status": "REALIZED_MEASUREMENTS_APPLIED", "accepted": result["accepted"]}, indent=1))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--phase-a", action="store_true", help="Build scorer-light Phase A receipts (default).")
    phase.add_argument("--phase-b", action="store_true", help="Run/queue Phase B gate handling.")
    parser.add_argument("--require-jd5", action="store_true", help="Require a complete jd5 endpoint receipt before Phase B.")
    parser.add_argument("--realized-measurement-jsonl", type=Path, default=None)
    parser.add_argument("--base-sub", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit"))
    parser.add_argument("--token-source", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_br1_20260803/cx1_tokens.npy"))
    parser.add_argument("--gt-argmax", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/gt_argmax_n600.npy"))
    parser.add_argument("--cx1-argmax", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/cx1_argmax_n600.npy"))
    parser.add_argument("--g3-jsonl", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_g3_score_atlas_n600_20260722T204000Z/ddm_g3_score_atlas_n600.jsonl"))
    parser.add_argument("--g4-recurrence", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_g4_spatial_stationarity_n600_20260722T212138Z/stage_checkpoints/01_recurrence_arrays.npz"))
    parser.add_argument("--sg1-cell-flip-mass", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_sg1_20260731/cell_flip_mass.npy"))
    parser.add_argument("--out-root", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/optimal_form"))
    parser.add_argument("--receipt-dir", type=Path, default=REPO / ".omx/research/ddm_tq1_20260805")
    parser.add_argument("--jd5-dir", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_jd4_20260805"))
    parser.add_argument("--price-top", type=int, default=12)
    parser.add_argument("--smoke-moves", type=int, default=4)
    args = parser.parse_args(argv)
    if not args.phase_a and not args.phase_b:
        args.phase_a = True
    if args.price_top < 1:
        raise SystemExit("--price-top must be positive")
    if args.smoke_moves < 0:
        raise SystemExit("--smoke-moves must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase_b:
        return run_phase_b(args)
    return run_phase_a(args)


if __name__ == "__main__":
    raise SystemExit(main())
