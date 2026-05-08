#!/usr/bin/env python3
"""Plan PR101 architecture-shrink and rate-shaped retraining candidates.

This is a deterministic CPU-only planning tool. It converts the current PR101
decoder substrate into a small set of explicit architecture / sparsity /
quantization targets, estimates the byte floor for each target, and emits the
training-driver manifest fields needed before GPU work is dispatchable.

The estimates are not score claims. They are bounded rate-side predictions
derived from measured PR101 symbols, the existing PR101 entropy-floor ladder,
and operator-supplied shrink/quantization targets. Exact contest score remains
CUDA auth eval on a produced archive.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.hnerv_arch_schema import (  # noqa: E402
    generate_hnerv_state_schema,
    schema_fingerprint,
    schema_numel,
    select_base_channels_for_element_retention,
)
from tac.pr101_split_brotli_codec import (  # noqa: E402
    BASE_CHANNELS,
    EVAL_SIZE,
    FIXED_STATE_SCHEMA,
    LATENT_DIM,
    N_QUANT,
    _quantize_tensor,
    encode_decoder_compact,
)

TOOL_NAME = "tools/plan_pr101_arch_shrink_retraining.py"
SCHEMA_VERSION = "pr101_arch_shrink_retraining_plan.v1"
ORIGINAL_VIDEO_BYTES = 37_545_489
DEFAULT_STATE_DICT = Path(
    "experiments/results/cma_pr101_real_substrate_20260507T222605Z/"
    "pr101_decoder_state_dict.pt"
)
DEFAULT_ENTROPY_REPORT = Path("reports/pr101_provable_optimal_floor.json")
DEFAULT_OUTPUT_DIR = Path("experiments/results/pr101_arch_shrink_retraining_plan_20260507_worker_b")
REFERENCE_ARCHIVE_BYTES = 178_144
REFERENCE_PAYLOAD_BYTES = 162_050
REFERENCE_ARCHIVE_OVERHEAD_BYTES = REFERENCE_ARCHIVE_BYTES - REFERENCE_PAYLOAD_BYTES


@dataclass(frozen=True)
class Scenario:
    """One architecture-shrink planning row.

    element_retention is the fraction of baseline decoder weight elements that
    remain after structured architecture shrink. sparsity is an additional
    fraction removed by pruning within that smaller architecture. quant_bits is
    the target stored precision. entropy_ratio is the predicted multiplier on
    current per-tensor H0 after rate-shaped retraining; values below 1.0 mean
    the trained weights are predicted to be lower-entropy than today's PR101
    quantized substrate.
    """

    name: str
    element_retention: float
    quant_bits: float
    entropy_ratio: float
    sparsity: float = 0.0
    side_info_bytes: int = 0
    sparse_mask_mode: str = "none"
    driver_family: str = "hnerv_arch_shrink_deltaepszeta"
    notes: str = ""

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> Scenario:
        allowed = {field.name for field in dataclasses.fields(cls)}
        extra = sorted(set(raw) - allowed)
        if extra:
            raise ValueError(f"scenario {raw.get('name', '<unnamed>')} has unknown keys: {extra}")
        return cls(**raw)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("scenario name must be non-empty")
        if not (0.0 < self.element_retention <= 1.0):
            raise ValueError(
                f"scenario {self.name}: element_retention must be in (0, 1], "
                f"got {self.element_retention}"
            )
        if not (0.0 <= self.sparsity < 1.0):
            raise ValueError(
                f"scenario {self.name}: sparsity must be in [0, 1), got {self.sparsity}"
            )
        if not (0.0 < self.quant_bits <= 8.0):
            raise ValueError(
                f"scenario {self.name}: quant_bits must be in (0, 8], got {self.quant_bits}"
            )
        if not (0.0 < self.entropy_ratio <= 1.5):
            raise ValueError(
                f"scenario {self.name}: entropy_ratio must be in (0, 1.5], "
                f"got {self.entropy_ratio}"
            )
        if self.side_info_bytes < 0:
            raise ValueError(
                f"scenario {self.name}: side_info_bytes must be >= 0, "
                f"got {self.side_info_bytes}"
            )
        if self.sparse_mask_mode not in {
            "none",
            "structured_channel_manifest",
            "unstructured_bitmask_brotli_estimate",
        }:
            raise ValueError(
                f"scenario {self.name}: unsupported sparse_mask_mode "
                f"{self.sparse_mask_mode!r}"
            )


def default_scenarios() -> list[Scenario]:
    """Return bounded default targets for the first GPU design pass."""

    return [
        Scenario(
            name="control_current_pr101_int8",
            element_retention=1.0,
            quant_bits=8.0,
            entropy_ratio=1.0,
            notes="Measured current-substrate control; no architecture shrink.",
        ),
        Scenario(
            name="stage_a_width090_int8_rate_shape",
            element_retention=0.81,
            quant_bits=8.0,
            entropy_ratio=0.98,
            side_info_bytes=256,
            notes="Conservative width shrink; target keeps PR101 topology class.",
        ),
        Scenario(
            name="stage_b_width080_int8_dez",
            element_retention=0.64,
            quant_bits=8.0,
            entropy_ratio=0.96,
            side_info_bytes=512,
            notes="Moderate structured shrink with delta-epsilon-zeta rate shaping.",
        ),
        Scenario(
            name="stage_c_width075_int6_qat_dez",
            element_retention=0.56,
            quant_bits=6.0,
            entropy_ratio=0.94,
            side_info_bytes=1536,
            notes="Int6 QAT plus structured shrink; requires runtime schema manifest.",
        ),
        Scenario(
            name="stage_d_zeta_width_precision_int4",
            element_retention=0.45,
            sparsity=0.10,
            quant_bits=4.0,
            entropy_ratio=0.90,
            side_info_bytes=4096,
            sparse_mask_mode="structured_channel_manifest",
            driver_family="self_compress_width_precision_hnerv",
            notes="SCNN-style width x precision target; structured manifest only.",
        ),
        Scenario(
            name="stage_e_high_risk_int4_sparse",
            element_retention=0.42,
            sparsity=0.25,
            quant_bits=4.0,
            entropy_ratio=0.86,
            side_info_bytes=6144,
            sparse_mask_mode="unstructured_bitmask_brotli_estimate",
            driver_family="imp_plus_self_compress_hnerv",
            notes="High-risk IMP/self-compress target; mask overhead charged.",
        ),
    ]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entropy_bits(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total == 0.0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / total
    return float(-np.sum(p * np.log2(p)))


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"state_dict not found: {path}")
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise TypeError(f"loaded state_dict is not a dict: {type(state_dict)!r}")
    return state_dict


def _quantized_tensor_rows(state_dict: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (name, shape) in enumerate(FIXED_STATE_SCHEMA):
        if name not in state_dict:
            raise KeyError(f"state_dict missing tensor {name!r}")
        tensor = state_dict[name]
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(
                f"state_dict tensor {name!r} shape {tuple(tensor.shape)} != schema {shape}"
            )
        qt = _quantize_tensor(name, tensor, n_quant=N_QUANT)
        symbols = (qt.q_i8.astype(np.int16) + 127).astype(np.uint8).reshape(-1)
        counts = np.bincount(symbols.astype(np.int32), minlength=255).astype(np.float64)
        h0 = _entropy_bits(counts)
        rows.append(
            {
                "idx": idx,
                "name": name,
                "shape": list(shape),
                "n_symbols": int(symbols.size),
                "h0_bits_per_symbol": h0,
                "h0_floor_bytes": math.ceil(symbols.size * h0 / 8.0),
                "scale": float(qt.scale),
            }
        )
    return rows


def _read_entropy_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"entropy floor report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _floor_by_name(report: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if report is None:
        return None
    for row in report.get("provable_floors", []):
        if row.get("name") == name:
            return row
    return None


def _mask_overhead_bytes(
    scenario: Scenario,
    *,
    post_arch_symbols: int,
) -> int:
    if scenario.sparse_mask_mode == "none":
        return 0
    if scenario.sparse_mask_mode == "structured_channel_manifest":
        return 0
    # Prediction: unstructured bitmask compressed by Brotli. This is charged
    # because unstructured sparsity is not free at inflate time.
    raw_mask_bytes = math.ceil(post_arch_symbols / 8.0)
    return math.ceil(raw_mask_bytes * 0.35)


def _rate_score_delta(bytes_delta: int) -> float:
    return 25.0 * bytes_delta / ORIGINAL_VIDEO_BYTES


def _estimate_scenario(
    scenario: Scenario,
    *,
    n_total_symbols: int,
    iid_per_tensor_bits: float,
    current_payload_bytes: int,
    archive_overhead_bytes: int,
    reference_archive_bytes: int,
) -> dict[str, Any]:
    scenario.validate()
    target_config = select_base_channels_for_element_retention(
        element_retention=scenario.element_retention,
        latent_dim=LATENT_DIM,
        baseline_base_channels=BASE_CHANNELS,
        eval_size=EVAL_SIZE,
    )
    target_schema = generate_hnerv_state_schema(target_config)
    base_h0_bps = iid_per_tensor_bits / max(1, n_total_symbols)
    post_arch_symbols = math.ceil(n_total_symbols * scenario.element_retention)
    active_symbols = math.ceil(post_arch_symbols * (1.0 - scenario.sparsity))
    predicted_h0_bps = min(float(scenario.quant_bits), base_h0_bps * scenario.entropy_ratio)
    predicted_floor_payload = math.ceil(active_symbols * predicted_h0_bps / 8.0)
    current_floor_payload = math.ceil(iid_per_tensor_bits / 8.0)
    coding_efficiency = current_payload_bytes / max(1, current_floor_payload)
    mask_overhead = _mask_overhead_bytes(scenario, post_arch_symbols=post_arch_symbols)
    expected_payload = int(
        math.ceil(predicted_floor_payload * coding_efficiency)
        + scenario.side_info_bytes
        + mask_overhead
    )
    expected_archive = expected_payload + archive_overhead_bytes
    delta_bytes = expected_archive - reference_archive_bytes

    blocker_basis = [
        "no_trained_checkpoint_for_this_architecture",
        "no_runtime_decoder_schema_manifest_for_shrunk_HNeRV",
        "no_inflate_output_parity_or_score_delta_classification",
        "no_exact_cuda_auth_eval",
    ]
    if scenario.sparse_mask_mode == "unstructured_bitmask_brotli_estimate":
        blocker_basis.append("unstructured_sparsity_mask_overhead_is_prediction_only")
    if scenario.quant_bits < 8.0:
        blocker_basis.append("quantized_runtime_export_not_implemented_for_target_bits")

    evidence = "empirical" if scenario.name == "control_current_pr101_int8" else "prediction"
    return {
        "name": scenario.name,
        "evidence_grade": evidence,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "charged_bits_changed": False,
        "driver_family": scenario.driver_family,
        "targets": {
            "element_retention": scenario.element_retention,
            "post_arch_symbols": post_arch_symbols,
            "sparsity": scenario.sparsity,
            "active_symbols": active_symbols,
            "quant_bits": scenario.quant_bits,
            "entropy_ratio_to_current_h0": scenario.entropy_ratio,
            "predicted_h0_bits_per_active_symbol": predicted_h0_bps,
            "sparse_mask_mode": scenario.sparse_mask_mode,
            "generated_hnerv_base_channels": target_config.base_channels,
            "generated_hnerv_channels": list(target_config.channels),
            "generated_schema_numel": schema_numel(target_schema),
            "generated_schema_fingerprint": schema_fingerprint(target_schema),
        },
        "byte_estimate": {
            "predicted_floor_payload_bytes": predicted_floor_payload,
            "current_codec_efficiency_vs_iid_floor": coding_efficiency,
            "side_info_bytes": scenario.side_info_bytes,
            "mask_overhead_bytes": mask_overhead,
            "expected_payload_bytes": expected_payload,
            "archive_overhead_bytes": archive_overhead_bytes,
            "expected_archive_bytes": expected_archive,
            "delta_archive_bytes_vs_reference": delta_bytes,
            "rate_score_delta_vs_reference": _rate_score_delta(delta_bytes),
        },
        "training_driver_manifest_fields": _driver_manifest_fields(
            scenario,
            generated_base_channels=target_config.base_channels,
            generated_schema_fingerprint=schema_fingerprint(target_schema),
        ),
        "dispatch_blockers": blocker_basis,
        "notes": scenario.notes,
    }


def _driver_manifest_fields(
    scenario: Scenario,
    *,
    generated_base_channels: int,
    generated_schema_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema": "pr101_arch_shrink_retraining_run.v1",
        "score_claim": False,
        "target_modes": ["contest_exact_eval"],
        "source_substrate": {
            "family": "PR101 hnerv_ft_microcodec",
            "model": "HNeRVDecoder",
            "latent_dim": LATENT_DIM,
            "base_channels": BASE_CHANNELS,
            "eval_size": list(EVAL_SIZE),
            "fixed_state_schema": "src/tac/pr101_split_brotli_codec.py::FIXED_STATE_SCHEMA",
        },
        "architecture_target": {
            "element_retention": scenario.element_retention,
            "generated_base_channels": generated_base_channels,
            "generated_schema_fingerprint": generated_schema_fingerprint,
            "structured_sparsity": scenario.sparsity,
            "quant_bits": scenario.quant_bits,
            "entropy_ratio_target": scenario.entropy_ratio,
            "sparse_mask_mode": scenario.sparse_mask_mode,
            "requires_generated_schema": scenario.element_retention != 1.0,
        },
        "training": {
            "base_curriculum": (
                "public PR106 belt_and_suspenders eight-stage HNeRV curriculum; "
                "parameterize latent_dim/base_channels/schema before reuse"
            ),
            "required_loss_terms": [
                "segnet_distillation_or_exact_source_stage_loss",
                "posenet_first6_pose_loss",
                "deltaepszeta_rate_proxy_weighted_by_tensor_headroom",
                "qat_fake_quant_matching_target_bits",
            ],
            "optional_loss_terms": [
                "self_compress_width_precision_l0",
                "codec_pipeline_epoch_byte_callback",
            ],
            "reproducibility_fields": [
                "seed",
                "git_commit",
                "input_video_sha256",
                "stage_configs",
                "optimizer_state",
                "ema_decay",
                "best_epoch",
                "decoder_state_dict_sha256",
                "latents_sha256",
                "codec_pipeline_manifest_sha256",
            ],
        },
        "archive_gates": [
            "state_dict_roundtrip_through_target_codec",
            "inflate_output_parity_or_explicit_distortion_classification",
            "deterministic_archive_manifest",
            "pre_submission_compliance_check_strict",
            "lane_dispatch_claim_before_gpu_eval",
            "contest_cuda_auth_eval_before_any_score_claim",
        ],
    }


def build_plan(
    *,
    state_dict_path: Path,
    entropy_floor_report: Path | None,
    scenarios: list[Scenario] | None = None,
    started_at_utc: str | None = None,
    skip_compact_encode: bool = False,
) -> dict[str, Any]:
    state_dict_path = state_dict_path if state_dict_path.is_absolute() else REPO_ROOT / state_dict_path
    report_path = None
    if entropy_floor_report is not None:
        report_path = entropy_floor_report if entropy_floor_report.is_absolute() else REPO_ROOT / entropy_floor_report

    state_dict = _load_state_dict(state_dict_path)
    tensor_rows = _quantized_tensor_rows(state_dict)
    n_total_symbols = sum(row["n_symbols"] for row in tensor_rows)
    iid_per_tensor_bits_measured = sum(
        row["n_symbols"] * row["h0_bits_per_symbol"] for row in tensor_rows
    )
    compact_payload_bytes = (
        REFERENCE_PAYLOAD_BYTES
        if skip_compact_encode
        else len(encode_decoder_compact(state_dict, brotli_quality=11))
    )

    report = _read_entropy_report(report_path)
    iid_floor_report = _floor_by_name(report, "iid_per_tensor")
    markov1_report = _floor_by_name(report, "markov1_per_tensor")
    markov2_report = _floor_by_name(report, "markov2_per_tensor")
    reference_archive_bytes = int(
        report.get("empirical_encoders", [{}])[0].get("bytes_archive", REFERENCE_ARCHIVE_BYTES)
        if report
        else REFERENCE_ARCHIVE_BYTES
    )
    archive_overhead_bytes = int(
        report.get("archive_overhead_bytes", REFERENCE_ARCHIVE_OVERHEAD_BYTES)
        if report
        else REFERENCE_ARCHIVE_OVERHEAD_BYTES
    )

    if iid_floor_report is not None:
        iid_per_tensor_bits = float(iid_floor_report["bits"])
    else:
        iid_per_tensor_bits = float(iid_per_tensor_bits_measured)

    scenario_list = scenarios if scenarios is not None else default_scenarios()
    scenario_rows = [
        _estimate_scenario(
            scenario,
            n_total_symbols=n_total_symbols,
            iid_per_tensor_bits=iid_per_tensor_bits,
            current_payload_bytes=compact_payload_bytes,
            archive_overhead_bytes=archive_overhead_bytes,
            reference_archive_bytes=reference_archive_bytes,
        )
        for scenario in scenario_list
    ]

    scenario_rows.sort(
        key=lambda row: row["byte_estimate"]["expected_archive_bytes"]
    )

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "started_at_utc": started_at_utc,
        "score_claim": False,
        "charged_bits_changed": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_partition": {
            "empirical": [
                "state_dict tensor count, symbols, H0, and PR101 compact Brotli bytes"
            ],
            "derivation": [
                "Shannon H0 payload floor, codec efficiency ratio, and contest rate-score delta formula"
            ],
            "prediction": [
                "architecture shrink, sparsity, quantization, entropy-ratio, side-info, and mask-overhead rows"
            ],
        },
        "inputs": {
            "state_dict_path": _repo_rel(state_dict_path),
            "state_dict_sha256": _sha256_file(state_dict_path),
            "entropy_floor_report": _repo_rel(report_path) if report_path else None,
            "entropy_floor_report_sha256": _sha256_file(report_path) if report_path else None,
        },
        "baseline": {
            "n_tensors": len(tensor_rows),
            "n_total_symbols": n_total_symbols,
            "latent_dim": LATENT_DIM,
            "base_channels": BASE_CHANNELS,
            "eval_size": list(EVAL_SIZE),
            "current_compact_decoder_payload_bytes": compact_payload_bytes,
            "reference_archive_bytes": reference_archive_bytes,
            "archive_overhead_bytes": archive_overhead_bytes,
            "iid_per_tensor_floor_payload_bytes": math.ceil(iid_per_tensor_bits / 8.0),
            "iid_per_tensor_floor_bits": iid_per_tensor_bits,
            "markov1_floor_payload_bytes": markov1_report.get("bytes_payload") if markov1_report else None,
            "markov2_floor_payload_bytes": markov2_report.get("bytes_payload") if markov2_report else None,
            "codec_efficiency_vs_iid_floor": compact_payload_bytes
            / max(1, math.ceil(iid_per_tensor_bits / 8.0)),
        },
        "per_tensor_top_h0_bytes": sorted(
            tensor_rows,
            key=lambda row: row["h0_floor_bytes"],
            reverse=True,
        )[:12],
        "scenarios": scenario_rows,
        "operator_next_commands": operator_next_commands(),
        "gpu_dispatch_blockers": [
            "Parameterize HNeRVDecoder and train_stage so shrunk base_channels/latent_dim produce a generated state schema.",
            "Implement a target codec/runtime loader for the generated schema; current PR101 FIXED_STATE_SCHEMA is hardcoded.",
            "Run CPU roundtrip through codec pipeline and local inflate parity before any remote eval.",
            "Claim lane via tools/claim_lane_dispatch.py before GPU training/eval dispatch.",
            "Run exact CUDA auth eval on produced archive bytes before setting score_claim=true.",
        ],
    }


def _repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def operator_next_commands() -> list[str]:
    return [
        ".venv/bin/python tools/plan_pr101_arch_shrink_retraining.py "
        "--state-dict-path experiments/results/cma_pr101_real_substrate_20260507T222605Z/pr101_decoder_state_dict.pt "
        "--entropy-floor-report reports/pr101_provable_optimal_floor.json "
        "--output-dir experiments/results/pr101_arch_shrink_retraining_plan_20260507_worker_b",
        ".venv/bin/python tools/build_hnerv_arch_shrink_driver.py "
        "--source-state-dict experiments/results/cma_pr101_real_substrate_20260507T222605Z/pr101_decoder_state_dict.pt "
        "--element-retention 0.45 "
        "--scenario-name stage_d_zeta_width_precision_int4 "
        "--output-dir experiments/results/hnerv_arch_shrink_training_driver_stage_d_20260507_codex",
        ".venv/bin/python tools/run_deltaepszeta_training.py "
        "--state-dict experiments/results/hnerv_arch_shrink_training_driver_stage_d_20260507_codex/initial_state_dict.pt "
        "--n-epochs 1 --steps-per-epoch 2 "
        "--log-dir experiments/results/hnerv_arch_shrink_training_driver_stage_d_20260507_codex/deltaepszeta_cpu_sanity "
        "--run-label stage_d_generated_schema_sanity",
        ".venv/bin/python tools/claim_lane_dispatch.py claim "
        "--lane-id pr101_arch_shrink_deltaepszeta_gpu --status planned "
        "--notes 'blocked until generated HNeRV schema/runtime and CPU roundtrip land'",
    ]


def render_markdown(plan: dict[str, Any]) -> str:
    baseline = plan["baseline"]
    best = plan["scenarios"][0]
    lines = [
        "# PR101 Architecture-Shrink Retraining Plan - 2026-05-07 Worker B",
        "",
        "## Evidence Boundary",
        "",
        "- `score_claim=false`; no archive was produced or evaluated.",
        "- Empirical: PR101 state-dict symbol counts, H0, compact decoder bytes.",
        "- Derivation: Shannon H0 floor and contest rate-score arithmetic.",
        "- Prediction: shrink / sparsity / quantization / entropy-ratio rows.",
        "",
        "## Baseline",
        "",
        f"- State dict: `{plan['inputs']['state_dict_path']}`",
        f"- SHA-256: `{plan['inputs']['state_dict_sha256']}`",
        f"- Tensors: {baseline['n_tensors']}",
        f"- Quantized symbols: {baseline['n_total_symbols']:,}",
        f"- Compact PR101 decoder payload: {baseline['current_compact_decoder_payload_bytes']:,} B",
        f"- IID per-tensor floor payload: {baseline['iid_per_tensor_floor_payload_bytes']:,} B",
        f"- Codec efficiency vs IID floor: {baseline['codec_efficiency_vs_iid_floor']:.4f}",
        f"- Markov-1 oracle payload: {baseline['markov1_floor_payload_bytes']:,} B",
        f"- Markov-2 oracle payload: {baseline['markov2_floor_payload_bytes']:,} B",
        "",
        "## Scenario Ranking",
        "",
        "| rank | scenario | evidence | retention | sparsity | bits | entropy ratio | expected archive | delta bytes | rate delta | dispatch |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(plan["scenarios"], 1):
        targets = row["targets"]
        estimate = row["byte_estimate"]
        lines.append(
            f"| {rank} | `{row['name']}` | {row['evidence_grade']} | "
            f"{targets['element_retention']:.2f} | {targets['sparsity']:.2f} | "
            f"{targets['quant_bits']:.1f} | {targets['entropy_ratio_to_current_h0']:.2f} | "
            f"{estimate['expected_archive_bytes']:,} | "
            f"{estimate['delta_archive_bytes_vs_reference']:+,} | "
            f"{estimate['rate_score_delta_vs_reference']:+.6f} | false |"
        )
    lines.extend(
        [
            "",
            "## Best Predicted Row",
            "",
            f"`{best['name']}` is the smallest rate-side estimate, but it is still blocked "
            "from dispatch because no trained checkpoint, generated schema, runtime loader, "
            "or exact CUDA eval exists for that architecture.",
            "",
            "## Integration Points Found",
            "",
            "- Public PR101/PR106 `HNeRVDecoder` fixes `latent_dim=28`, `base_channels=36`, and a 28-tensor schema.",
            "- `src/tac/pr101_split_brotli_codec.py::FIXED_STATE_SCHEMA` is hardcoded to the current architecture.",
            "- `tools/run_deltaepszeta_training.py` is a state-dict CPU sanity loop, not a renderer/scorer training driver.",
            "- `src/tac/self_compressing_nn.py` has width x precision accounting that can inform the loss once the HNeRV driver exists.",
            "- `src/tac/codec_pipeline_deltaepszeta_callback.py` can log codec bytes per epoch after a pipeline for the generated schema exists.",
            "",
            "## Operator Commands",
            "",
        ]
    )
    for command in plan["operator_next_commands"]:
        lines.extend(["```bash", command, "```", ""])
    lines.extend(["## GPU Dispatch Blockers", ""])
    for blocker in plan["gpu_dispatch_blockers"]:
        lines.append(f"- {blocker}")
    return "\n".join(lines).rstrip() + "\n"


def _load_scenarios(path: Path | None) -> list[Scenario] | None:
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("scenario JSON must be a list of objects")
    return [Scenario.from_mapping(item) for item in raw]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict-path", type=Path, default=DEFAULT_STATE_DICT)
    parser.add_argument("--entropy-floor-report", type=Path, default=DEFAULT_ENTROPY_REPORT)
    parser.add_argument("--scenario-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--started-at-utc",
        default="2026-05-07T00:00:00Z",
        help="fixed timestamp for byte-reproducible manifests",
    )
    parser.add_argument(
        "--skip-compact-encode",
        action="store_true",
        help="use the reference payload bytes instead of recomputing PR101 compact Brotli",
    )
    args = parser.parse_args(argv)

    scenarios = _load_scenarios(args.scenario_json)
    plan = build_plan(
        state_dict_path=args.state_dict_path,
        entropy_floor_report=args.entropy_floor_report,
        scenarios=scenarios,
        started_at_utc=args.started_at_utc,
        skip_compact_encode=args.skip_compact_encode,
    )

    out_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "plan.json"
    md_path = out_dir / "plan.md"
    json_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(plan), encoding="utf-8")

    print(f"wrote: {json_path}")
    print(f"wrote: {md_path}")
    print("best_predicted:")
    best = plan["scenarios"][0]
    estimate = best["byte_estimate"]
    print(f"  {best['name']}")
    print(f"  expected_archive_bytes={estimate['expected_archive_bytes']}")
    print(f"  delta_archive_bytes_vs_reference={estimate['delta_archive_bytes_vs_reference']:+d}")
    print(f"  rate_score_delta_vs_reference={estimate['rate_score_delta_vs_reference']:+.6f}")
    print("score_claim=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
