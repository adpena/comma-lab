#!/usr/bin/env python3
"""Plan Yousfi-Fridrich atom-field allocations from planning ledgers.

This tool has two linked outputs:

* ``contest``: a practical, contest-faithful atom-field allocator. It reads
  planning-only atom ledgers, prices row-run repair atoms with a Lagrangian
  rate term plus sparse curvature/interactions, and emits deterministic policy
  candidates that ``build_cmg3_adaptive_runs_candidate.py`` can consume.
* ``ideal``: an infinite-compute equation system for the complete archive
  optimization problem. It is recorded as math and search design only; it never
  claims score evidence or dispatches jobs.

Scores still require exact CUDA auth eval of the exact archive bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "yousfi_fridrich_atom_field_allocator_v1"
TOOL = "experiments/plan_yousfi_fridrich_field_equations.py"
ORIGINAL_VIDEO_BYTES = 37_545_489
LAMBDA_RATE = 25.0 / ORIGINAL_VIDEO_BYTES
DEFAULT_CANDIDATE_SIZES = (8, 16, 32, 64, 128, 256)
DEFAULT_MODE = "both"
DEFAULT_INTERACTION_MODEL = "sparse_pair_frame_class"


class FieldPlanError(ValueError):
    """Raised when field-equation planning inputs are invalid."""


@dataclass(frozen=True)
class Atom:
    source_ledger: str
    atom_id: str
    family: str
    identity: dict[str, Any]
    pair_indices: tuple[int, ...]
    frame_indices: tuple[int, ...]
    class_ids: tuple[int, ...]
    residual_pixels: int
    charged_bytes: int
    first_order_score_saved_proxy: float
    first_order_rate_cost: float
    first_order_net: float
    density: float
    weighted_residual_pixel_proxy: float
    expected_base_runs_per_row: int | None

    @property
    def row_run_key(self) -> tuple[int, int, int, int, int] | None:
        if self.family != "row_run":
            return None
        try:
            return (
                int(self.identity["frame_index"]),
                int(self.identity["y"]),
                int(self.identity["x0"]),
                int(self.identity["x1_exclusive"]),
                int(self.identity["class_id"]),
            )
        except KeyError as exc:
            raise FieldPlanError(f"row_run atom {self.atom_id} missing {exc.args[0]!r}") from exc

    def row_run_policy_atom(self) -> dict[str, int]:
        key = self.row_run_key
        if key is None:
            raise FieldPlanError(f"atom {self.atom_id} is not a row_run atom")
        return {
            "frame_index": key[0],
            "y": key[1],
            "x0": key[2],
            "x1_exclusive": key[3],
            "class_id": key[4],
        }


@dataclass(frozen=True)
class FieldConfig:
    mode: str = DEFAULT_MODE
    max_source_atoms: int = 512
    candidate_sizes: tuple[int, ...] = DEFAULT_CANDIDATE_SIZES
    interaction_model: str = DEFAULT_INTERACTION_MODEL
    curvature_strength: float = 0.08
    pair_antagonism: float = 1.0e-6
    frame_antagonism: float = 5.0e-7
    class_synergy: float = 5.0e-7
    low_rank_modes: int = 8
    positive_proxy_only: bool = False
    allow_negative_field_energy: bool = False
    policy_prefix: str = "yf_field"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FieldPlanError(f"{path} is not valid JSON") from exc


def _finite_float(value: Any, *, field: str, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FieldPlanError(f"{field} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise FieldPlanError(f"{field} must be finite")
    return out


def _int_tuple(values: Any, *, field: str) -> tuple[int, ...]:
    if values is None:
        return ()
    if not isinstance(values, list):
        raise FieldPlanError(f"{field} must be a list")
    out: list[int] = []
    for index, value in enumerate(values):
        if not isinstance(value, int):
            raise FieldPlanError(f"{field}[{index}] must be int")
        out.append(int(value))
    return tuple(out)


def _identity_int(identity: dict[str, Any], key: str) -> tuple[int, ...]:
    if key not in identity:
        return ()
    value = identity[key]
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, list):
        return tuple(int(v) for v in value if isinstance(v, int))
    return ()


def _atom_from_json(raw: dict[str, Any], *, source_ledger: str) -> Atom:
    if raw.get("score_claim") is True:
        raise FieldPlanError(f"atom {raw.get('atom_id')} has score_claim=true")
    identity = raw.get("identity")
    if not isinstance(identity, dict):
        raise FieldPlanError(f"atom {raw.get('atom_id')} has no identity object")
    cost_model = raw.get("cost_model")
    lagrangian = raw.get("lagrangian")
    weights = raw.get("weights", {})
    if not isinstance(cost_model, dict) or not isinstance(lagrangian, dict):
        raise FieldPlanError(f"atom {raw.get('atom_id')} lacks cost_model/lagrangian")
    charged = int(cost_model.get("estimated_charged_bytes", 0))
    if charged <= 0:
        raise FieldPlanError(f"atom {raw.get('atom_id')} has nonpositive charged byte estimate")
    benefit = _finite_float(
        lagrangian.get("estimated_marginal_score_saved_proxy"),
        field="estimated_marginal_score_saved_proxy",
    )
    rate_cost = _finite_float(
        lagrangian.get("estimated_rate_score_cost"),
        field="estimated_rate_score_cost",
        default=LAMBDA_RATE * charged,
    )
    net = _finite_float(
        lagrangian.get("estimated_lagrangian_net_proxy"),
        field="estimated_lagrangian_net_proxy",
        default=benefit - rate_cost,
    )
    density = _finite_float(
        lagrangian.get("estimated_score_saved_per_charged_byte"),
        field="estimated_score_saved_per_charged_byte",
        default=benefit / charged,
    )
    pair_indices = _int_tuple(raw.get("pair_indices"), field="pair_indices") or _identity_int(identity, "pair_index")
    frame_indices = _int_tuple(raw.get("frame_indices"), field="frame_indices") or _identity_int(identity, "frame_index")
    class_ids = _int_tuple(raw.get("class_ids"), field="class_ids") or _identity_int(identity, "class_id")
    return Atom(
        source_ledger=source_ledger,
        atom_id=str(raw.get("atom_id")),
        family=str(raw.get("atom_family")),
        identity=identity,
        pair_indices=tuple(sorted(set(pair_indices))),
        frame_indices=tuple(sorted(set(frame_indices))),
        class_ids=tuple(sorted(set(class_ids))),
        residual_pixels=int(raw.get("residual_pixels", 0)),
        charged_bytes=charged,
        first_order_score_saved_proxy=benefit,
        first_order_rate_cost=rate_cost,
        first_order_net=net,
        density=density,
        weighted_residual_pixel_proxy=_finite_float(
            weights.get("weighted_residual_pixel_proxy") if isinstance(weights, dict) else None,
            field="weighted_residual_pixel_proxy",
            default=float(raw.get("residual_pixels", 0)),
        ),
        expected_base_runs_per_row=None,
    )


def _candidate_expected_base_runs_per_row(payload: dict[str, Any]) -> int | None:
    candidate = payload.get("inputs", {}).get("candidate", {}) if isinstance(payload.get("inputs"), dict) else {}
    if not isinstance(candidate, dict):
        return None
    if candidate.get("mode") == "reconstructed_from_cmg3_nonzero_row_runs_manifest":
        raw = candidate.get("max_runs_per_row")
        return int(raw) if isinstance(raw, int) else None
    if candidate.get("mode") == "reconstructed_from_cmg3a_adaptive_manifest":
        raw = candidate.get("base_runs_per_row")
        return int(raw) if isinstance(raw, int) else None
    return None


def _load_atoms(ledger_paths: Iterable[Path], *, max_source_atoms: int) -> tuple[list[Atom], list[dict[str, Any]]]:
    atoms: list[Atom] = []
    inputs: list[dict[str, Any]] = []
    for path in ledger_paths:
        path = path.resolve()
        payload = _read_json(path)
        if not isinstance(payload, dict):
            raise FieldPlanError(f"{path} must contain a JSON object")
        if payload.get("score_claim") is True:
            raise FieldPlanError(f"{path} has score_claim=true; expected planning-only ledger")
        raw_atoms = payload.get("top_atoms")
        if not isinstance(raw_atoms, list):
            raise FieldPlanError(f"{path} must contain top_atoms")
        ledger_id = path.stem
        expected_base_runs_per_row = _candidate_expected_base_runs_per_row(payload)
        for raw in raw_atoms[:max_source_atoms]:
            if not isinstance(raw, dict):
                raise FieldPlanError(f"{path}: top_atoms entries must be objects")
            atom = _atom_from_json(raw, source_ledger=ledger_id)
            atoms.append(replace(atom, expected_base_runs_per_row=expected_base_runs_per_row))
        inputs.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "schema": payload.get("schema"),
                "evidence_grade": payload.get("evidence_grade"),
                "score_claim": payload.get("score_claim"),
                "atom_count": payload.get("atom_count"),
                "top_atoms_read": min(len(raw_atoms), max_source_atoms),
                "tensor": payload.get("tensor"),
                "expected_builder_base_runs_per_row": expected_base_runs_per_row,
            }
        )
    return atoms, inputs


def _dedupe_row_run_atoms(atoms: list[Atom], *, positive_proxy_only: bool) -> list[Atom]:
    best: dict[tuple[int, int, int, int, int], Atom] = {}
    for atom in atoms:
        key = atom.row_run_key
        if key is None:
            continue
        # CMG3A's nonzero-row-run wire grammar stores foreground class runs
        # over an implicit zero background. Class-0 atoms are valid analysis
        # signal, but they are not representable by this builder contract.
        if key[4] <= 0:
            continue
        if positive_proxy_only and atom.first_order_net <= 0.0:
            continue
        prev = best.get(key)
        if prev is None or _static_rank(atom) < _static_rank(prev):
            best[key] = atom
    return sorted(best.values(), key=_static_rank)


def _static_rank(atom: Atom) -> tuple[float, float, int, int, str]:
    return (
        -float(atom.density),
        -float(atom.first_order_net),
        -int(atom.residual_pixels),
        int(atom.charged_bytes),
        atom.atom_id,
    )


def _marginal_field_energy(
    atom: Atom,
    *,
    pair_load: Counter[int],
    frame_load: Counter[int],
    class_load: Counter[int],
    config: FieldConfig,
) -> dict[str, float]:
    benefit = atom.first_order_score_saved_proxy
    rate = LAMBDA_RATE * atom.charged_bytes
    pair_cost = sum(2.0 * pair_load[pair] + 1.0 for pair in atom.pair_indices)
    frame_cost = sum(2.0 * frame_load[frame] + 1.0 for frame in atom.frame_indices)
    class_gain = sum(1.0 / math.sqrt(class_load[class_id] + 1.0) for class_id in atom.class_ids)
    curvature = config.curvature_strength * (atom.density * atom.density) * atom.charged_bytes
    antagonism = config.pair_antagonism * pair_cost + config.frame_antagonism * frame_cost
    synergy = config.class_synergy * class_gain
    total = benefit - rate - curvature - antagonism + synergy
    return {
        "first_order_score_saved_proxy": benefit,
        "rate_cost": rate,
        "curvature_penalty": curvature,
        "pair_frame_antagonism": antagonism,
        "class_synergy": synergy,
        "marginal_field_energy": total,
    }


def _greedy_candidates(
    row_atoms: list[Atom],
    *,
    config: FieldConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    max_size = max(config.candidate_sizes)
    remaining = list(row_atoms)
    selected: list[Atom] = []
    pair_load: Counter[int] = Counter()
    frame_load: Counter[int] = Counter()
    class_load: Counter[int] = Counter()
    candidate_by_size: dict[int, dict[str, Any]] = {}
    filtered_negative: list[dict[str, Any]] = []
    totals = {
        "first_order_score_saved_proxy": 0.0,
        "rate_cost": 0.0,
        "curvature_penalty": 0.0,
        "pair_frame_antagonism": 0.0,
        "class_synergy": 0.0,
        "marginal_field_energy": 0.0,
    }

    for step in range(min(max_size, len(remaining))):
        ranked = []
        for atom in remaining:
            terms = _marginal_field_energy(
                atom,
                pair_load=pair_load,
                frame_load=frame_load,
                class_load=class_load,
                config=config,
            )
            ranked.append(
                (
                    -terms["marginal_field_energy"],
                    _static_rank(atom),
                    atom,
                    terms,
                )
            )
        ranked.sort(key=lambda item: (item[0], item[1]))
        _neg_energy, _rank, chosen, terms = ranked[0]
        remaining.remove(chosen)
        selected.append(chosen)
        for pair in chosen.pair_indices:
            pair_load[pair] += 1
        for frame in chosen.frame_indices:
            frame_load[frame] += 1
        for class_id in chosen.class_ids:
            class_load[class_id] += 1
        for key in totals:
            totals[key] += float(terms[key])
        current_size = step + 1
        if current_size in config.candidate_sizes:
            policy = _policy_from_selection(
                selected,
                config=config,
                selection_size=current_size,
                totals=totals,
                pair_load=pair_load,
                frame_load=frame_load,
                class_load=class_load,
            )
            if (
                config.allow_negative_field_energy
                or float(policy["estimated_proxy"]["field_energy"]) > 0.0
            ):
                candidate_by_size[current_size] = policy
            else:
                filtered_negative.append(
                    {
                        "policy_id": policy["policy_id"],
                        "selected_atom_count": policy["selected_atom_count"],
                        "field_energy": policy["estimated_proxy"]["field_energy"],
                        "reason": "negative_field_energy_filtered_by_default",
                    }
                )
    return (
        [candidate_by_size[size] for size in config.candidate_sizes if size in candidate_by_size],
        filtered_negative,
    )


def _policy_from_selection(
    selected: list[Atom],
    *,
    config: FieldConfig,
    selection_size: int,
    totals: dict[str, float],
    pair_load: Counter[int],
    frame_load: Counter[int],
    class_load: Counter[int],
) -> dict[str, Any]:
    selected_copy = list(selected)
    selected_bytes = sum(atom.charged_bytes for atom in selected_copy)
    selected_pixels = sum(atom.residual_pixels for atom in selected_copy)
    policy_id = f"{config.policy_prefix}_{config.interaction_model}_top{selection_size:04d}"
    top_pairs = [pair for pair, _count in pair_load.most_common(24)]
    top_frames = [frame for frame, _count in frame_load.most_common(24)]
    expected_bases = sorted(
        {
            int(atom.expected_base_runs_per_row)
            for atom in selected_copy
            if atom.expected_base_runs_per_row is not None
        }
    )
    expected_base = expected_bases[0] if len(expected_bases) == 1 else None
    return {
        "policy_id": policy_id,
        "mode": "contest_practical_field_equation",
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "planning_only",
        "builder": "experiments/build_cmg3_adaptive_runs_candidate.py --field-policy-json <this-json> --field-policy-id "
        + policy_id,
        "selected_atom_count": len(selected_copy),
        "selected_row_run_atoms": [atom.row_run_policy_atom() for atom in selected_copy],
        "selected_atom_ids": [atom.atom_id for atom in selected_copy],
        "source_ledgers": sorted({atom.source_ledger for atom in selected_copy}),
        "required_base_runs_per_row": expected_base,
        "expected_base_runs_per_row": expected_base,
        "expected_base_runs_per_row_set": expected_bases,
        "estimated_proxy": {
            "selected_uncompressed_proxy_bytes": int(selected_bytes),
            "selected_residual_pixels": int(selected_pixels),
            "first_order_score_saved_proxy": round(totals["first_order_score_saved_proxy"], 12),
            "rate_score_cost": round(totals["rate_cost"], 12),
            "curvature_penalty": round(totals["curvature_penalty"], 12),
            "pair_frame_antagonism": round(totals["pair_frame_antagonism"], 12),
            "class_synergy": round(totals["class_synergy"], 12),
            "field_energy": round(totals["marginal_field_energy"], 12),
            "density_proxy": round(totals["marginal_field_energy"] / max(selected_bytes, 1), 12),
        },
        "support": {
            "top_pair_indices_by_selected_atom_count": top_pairs,
            "top_frame_indices_by_selected_atom_count": top_frames,
            "class_atom_counts": {str(k): int(class_load[k]) for k in sorted(class_load)},
        },
        "required_next_steps": [
            "build concrete archive with this exact policy json, policy id, and expected base-runs semantics",
            "measure archive bytes and SHA",
            "run exact CUDA diagnostic auth eval",
            "rerun identical archive bytes on T4/equivalent only if diagnostic survives component gates",
        ],
    }


def _contest_equations(config: FieldConfig) -> dict[str, Any]:
    return {
        "name": "contest_practical_atom_field_equations",
        "variables": {
            "x_a": "binary decision for charged archive atom a",
            "c_a": "charged byte cost of atom a inside archive",
            "b_a": "first-order scorer benefit proxy from exact residual/component evidence",
            "H_ab": "sparse sampled interaction or curvature between atoms a,b",
            "lambda_rate": "25 / 37545489",
        },
        "objective": (
            "maximize F(x) = sum_a b_a*x_a - lambda_rate*sum_a c_a*x_a "
            "- 0.5*sum_{a,b} H_ab*x_a*x_b + sum_g synergy_g(x)"
        ),
        "constraints": [
            "all score-affecting bits must be inside archive.zip",
            "archive must inflate deterministically through fixed contest runtime",
            "runtime must pass T4-equivalent budget before promotion",
            "selected atoms become score evidence only after exact CUDA auth eval",
            "stacked deltas are not additive evidence until the stacked archive is exact-evaluated",
        ],
        "implemented_sparse_surrogate": {
            "interaction_model": config.interaction_model,
            "marginal_energy": (
                "DeltaF(a|S)=b_a-lambda*c_a-curvature(a)-pair_frame_antagonism(a,S)+class_synergy(a,S)"
            ),
            "curvature_strength": config.curvature_strength,
            "pair_antagonism": config.pair_antagonism,
            "frame_antagonism": config.frame_antagonism,
            "class_synergy": config.class_synergy,
        },
    }


def _ideal_equations(config: FieldConfig) -> dict[str, Any]:
    return {
        "name": "ideal_infinite_compute_archive_field_equations",
        "status": "mathematical_complete_system_not_score_evidence",
        "variables": {
            "A": "complete archive byte string including decoder and all payload bits",
            "D_A": "deterministic inflate map induced by archive A",
            "M_A,P_A": "inflated masks and poses/rendered frames consumed by the scorer",
            "theta": "decoder/model/grammar/quantizer parameters charged inside A",
            "z": "latent or side-information bits charged inside A",
        },
        "exact_objective": (
            "min_A 100*seg_dist(M_A) + sqrt(10*pose_dist(P_A)) + 25*|A|/37545489"
        ),
        "field_form": (
            "delta S = integral g_i dphi_i + 1/2 integral integral H_ij dphi_i dphi_j + "
            "higher_order_terms, with phi spanning pixels, runs, poses, latents, decoder weights, and packer bits"
        ),
        "all_order_expansion": [
            "Taylor/Frechet: exact local scorer expansion over atom coordinates",
            "Fourier/Walsh: randomized low-dimensional subspace probes recover sparse interaction coefficients",
            "Riemannian: optimize on pose/camera/ego-motion manifolds instead of arbitrary pixel axes",
            "Feynman/CEM: sample correction paths, weight by exact/proxy energy, refit proposal distribution",
            "Dykstra/ADMM: project against rate, distortion, runtime, compliance, and reproducibility constraints",
        ],
        "infinite_compute_search": [
            "enumerate or sample archive grammars and decoder families",
            "for every candidate atom set, build exact archive bytes",
            "run exact CUDA auth eval as the oracle",
            "fit all-order interaction tensor from oracle calls",
            "solve the charged archive minimum and promote only identical bytes on T4/equivalent",
        ],
        "practical_projection": (
            "The contest mode is the computable low-order projection of this system under the current wall-clock budget."
        ),
        "low_rank_modes": config.low_rank_modes,
    }


def build_plan(
    *,
    ledger_jsons: list[Path],
    output_json: Path,
    mode: str = DEFAULT_MODE,
    max_source_atoms: int = 512,
    candidate_sizes: tuple[int, ...] = DEFAULT_CANDIDATE_SIZES,
    interaction_model: str = DEFAULT_INTERACTION_MODEL,
    curvature_strength: float = 0.08,
    pair_antagonism: float = 1.0e-6,
    frame_antagonism: float = 5.0e-7,
    class_synergy: float = 5.0e-7,
    low_rank_modes: int = 8,
    positive_proxy_only: bool = False,
    allow_negative_field_energy: bool = False,
    policy_prefix: str = "yf_field",
) -> dict[str, Any]:
    if mode not in {"contest", "ideal", "both"}:
        raise FieldPlanError("mode must be contest, ideal, or both")
    if not ledger_jsons:
        raise FieldPlanError("at least one --ledger-json is required")
    if max_source_atoms <= 0:
        raise FieldPlanError("max_source_atoms must be positive")
    if not candidate_sizes or any(size <= 0 for size in candidate_sizes):
        raise FieldPlanError("candidate_sizes must contain positive integers")
    if low_rank_modes <= 0:
        raise FieldPlanError("low_rank_modes must be positive")
    for name, value in {
        "curvature_strength": curvature_strength,
        "pair_antagonism": pair_antagonism,
        "frame_antagonism": frame_antagonism,
        "class_synergy": class_synergy,
    }.items():
        if value < 0.0:
            raise FieldPlanError(f"{name} must be nonnegative")

    config = FieldConfig(
        mode=mode,
        max_source_atoms=max_source_atoms,
        candidate_sizes=tuple(sorted(set(int(v) for v in candidate_sizes))),
        interaction_model=interaction_model,
        curvature_strength=curvature_strength,
        pair_antagonism=pair_antagonism,
        frame_antagonism=frame_antagonism,
        class_synergy=class_synergy,
        low_rank_modes=low_rank_modes,
        positive_proxy_only=positive_proxy_only,
        allow_negative_field_energy=allow_negative_field_energy,
        policy_prefix=policy_prefix,
    )
    atoms, inputs = _load_atoms(ledger_jsons, max_source_atoms=max_source_atoms)
    row_atoms = _dedupe_row_run_atoms(atoms, positive_proxy_only=positive_proxy_only)
    if mode in {"contest", "both"}:
        policies, filtered_negative = _greedy_candidates(row_atoms, config=config)
    else:
        policies, filtered_negative = [], []
    family_counts = Counter(atom.family for atom in atoms)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "producer": TOOL,
        "mode": mode,
        "score_claim": False,
        "no_score_claim": True,
        "promotion_eligible": False,
        "evidence_grade": "planning_only",
        "cuda_jobs_launched": False,
        "remote_jobs_dispatched": False,
        "canonical_score_source_required": (
            "archive.zip -> inflate.sh -> upstream/evaluate.py via "
            "experiments/contest_auth_eval.py --device cuda"
        ),
        "configuration": {
            "max_source_atoms": max_source_atoms,
            "candidate_sizes": list(config.candidate_sizes),
            "interaction_model": interaction_model,
            "curvature_strength": curvature_strength,
            "pair_antagonism": pair_antagonism,
            "frame_antagonism": frame_antagonism,
            "class_synergy": class_synergy,
            "low_rank_modes": low_rank_modes,
            "positive_proxy_only": positive_proxy_only,
            "allow_negative_field_energy": allow_negative_field_energy,
            "policy_prefix": policy_prefix,
            "env_defaults": {
                "PACT_FIELD_EQUATION_MODE": os.environ.get("PACT_FIELD_EQUATION_MODE"),
                "PACT_FIELD_CANDIDATE_SIZES": os.environ.get("PACT_FIELD_CANDIDATE_SIZES"),
                "PACT_FIELD_MAX_SOURCE_ATOMS": os.environ.get("PACT_FIELD_MAX_SOURCE_ATOMS"),
                "PACT_FIELD_INTERACTION_MODEL": os.environ.get("PACT_FIELD_INTERACTION_MODEL"),
                "PACT_FIELD_ALLOW_NEGATIVE_FIELD_ENERGY": os.environ.get("PACT_FIELD_ALLOW_NEGATIVE_FIELD_ENERGY"),
            },
        },
        "inputs": inputs,
        "formulas": {
            "contest_score_formula": (
                "score = 100*seg_dist + sqrt(10*pose_dist) + 25*archive_bytes/37545489"
            ),
            "lambda_rate": LAMBDA_RATE,
            "lambda_rate_formula": "25 / 37545489",
            "break_even_bytes_per_score": ORIGINAL_VIDEO_BYTES / 25.0,
        },
        "contest_practical_equations": _contest_equations(config)
        if mode in {"contest", "both"}
        else None,
        "ideal_infinite_compute_equations": _ideal_equations(config)
        if mode in {"ideal", "both"}
        else None,
        "atom_summary": {
            "source_atom_count": len(atoms),
            "row_run_atom_count": len([atom for atom in atoms if atom.family == "row_run"]),
            "deduped_row_run_atom_count": len(row_atoms),
            "positive_row_run_net_count": len([atom for atom in row_atoms if atom.first_order_net > 0.0]),
            "atom_family_counts": {str(k): int(family_counts[k]) for k in sorted(family_counts)},
        },
        "candidate_policies": policies,
        "filtered_candidate_policies": {
            "negative_field_energy": filtered_negative,
            "negative_field_energy_policy": (
                "filtered by default; pass --allow-negative-field-energy only for explicit cliff-mapping diagnostics"
            ),
        },
        "required_next_steps": [
            "build concrete CMG3A archives from candidate_policies using --field-policy-json",
            "byte-screen built archives and discard byte-regressive or high-disagreement variants",
            "run fast CUDA diagnostic exact eval on the Pareto set",
            "run T4/equivalent promotion only on identical bytes that survive diagnostics",
        ],
    }
    _write_json(output_json, payload)
    return payload


def _parse_int_list(value: str) -> tuple[int, ...]:
    out = []
    for raw in value.split(","):
        raw = raw.strip()
        if raw:
            out.append(int(raw))
    if not out:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    if any(v <= 0 for v in out):
        raise argparse.ArgumentTypeError("candidate sizes must be positive")
    return tuple(out)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-json", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("contest", "ideal", "both"),
        default=os.environ.get("PACT_FIELD_EQUATION_MODE", DEFAULT_MODE),
    )
    parser.add_argument("--max-source-atoms", type=int, default=_env_int("PACT_FIELD_MAX_SOURCE_ATOMS", 512))
    parser.add_argument(
        "--candidate-sizes",
        type=_parse_int_list,
        default=_parse_int_list(os.environ.get("PACT_FIELD_CANDIDATE_SIZES", "8,16,32,64,128,256")),
    )
    parser.add_argument(
        "--interaction-model",
        default=os.environ.get("PACT_FIELD_INTERACTION_MODEL", DEFAULT_INTERACTION_MODEL),
    )
    parser.add_argument("--curvature-strength", type=float, default=_env_float("PACT_FIELD_CURVATURE_STRENGTH", 0.08))
    parser.add_argument("--pair-antagonism", type=float, default=_env_float("PACT_FIELD_PAIR_ANTAGONISM", 1.0e-6))
    parser.add_argument("--frame-antagonism", type=float, default=_env_float("PACT_FIELD_FRAME_ANTAGONISM", 5.0e-7))
    parser.add_argument("--class-synergy", type=float, default=_env_float("PACT_FIELD_CLASS_SYNERGY", 5.0e-7))
    parser.add_argument("--low-rank-modes", type=int, default=_env_int("PACT_FIELD_LOW_RANK_MODES", 8))
    parser.add_argument(
        "--positive-proxy-only",
        action="store_true",
        default=_env_bool("PACT_FIELD_POSITIVE_PROXY_ONLY", False),
    )
    parser.add_argument(
        "--allow-negative-field-energy",
        action="store_true",
        default=_env_bool("PACT_FIELD_ALLOW_NEGATIVE_FIELD_ENERGY", False),
        help="Emit negative-energy policies for explicit cliff-mapping diagnostics.",
    )
    parser.add_argument("--policy-prefix", default=os.environ.get("PACT_FIELD_POLICY_PREFIX", "yf_field"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_plan(
        ledger_jsons=args.ledger_json,
        output_json=args.output_json,
        mode=args.mode,
        max_source_atoms=args.max_source_atoms,
        candidate_sizes=args.candidate_sizes,
        interaction_model=args.interaction_model,
        curvature_strength=args.curvature_strength,
        pair_antagonism=args.pair_antagonism,
        frame_antagonism=args.frame_antagonism,
        class_synergy=args.class_synergy,
        low_rank_modes=args.low_rank_modes,
        positive_proxy_only=bool(args.positive_proxy_only),
        allow_negative_field_energy=bool(args.allow_negative_field_energy),
        policy_prefix=args.policy_prefix,
    )
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "schema": payload["schema"],
                "mode": payload["mode"],
                "candidate_policy_count": len(payload["candidate_policies"]),
                "deduped_row_run_atom_count": payload["atom_summary"]["deduped_row_run_atom_count"],
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
