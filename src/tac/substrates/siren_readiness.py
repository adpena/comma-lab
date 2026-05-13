"""Fail-closed readiness audit for the SIREN substrate first anchor.

This module is intentionally text/manifest based. It must be cheap enough for
operator preflight and must not import torch, scorers, or provider SDKs. The
audit answers one narrow question: does the repository contain the local
trainer, recipe, archive grammar, and runtime surfaces required before an
operator can even consider a SIREN first-anchor training dispatch?
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

LANE_ID = "lane_substrate_siren_20260512"

REQUIRED_TRAINER_FLAGS = (
    "--video-path",
    "--output-dir",
    "--epochs",
    "--batch-size",
    "--upstream-dir",
    "--device",
    "--dispatch-contract",
)

VALID_TARGET_MODES = (
    "contest_exact_eval",
    "contest_one_video_replay",
    "contest_generalized",
    "production_generalized",
    "production_edge_adaptive",
    "research_substrate",
)

REQUIRED_DISPATCH_CONTRACTS = (
    "naked_siren_replacement",
    "siren_residual_on_hnerv_a1",
    "hybrid_siren_domain_prior",
)

REQUIRED_PATHS = {
    "trainer": Path("experiments/train_substrate_siren.py"),
    "recipe": Path(".omx/operator_authorize_recipes/substrate_siren_modal_a100_dispatch.yaml"),
    "architecture": Path("src/tac/substrates/siren/architecture.py"),
    "archive": Path("src/tac/substrates/siren/archive.py"),
    "inflate": Path("src/tac/substrates/siren/inflate.py"),
    "score_aware_loss": Path("src/tac/substrates/siren/score_aware_loss.py"),
    "roundtrip_tests": Path("src/tac/substrates/siren/tests/test_siren_roundtrip.py"),
    "score_loss_tests": Path(
        "src/tac/substrates/siren/tests/test_score_aware_loss_real_scorer_forward.py"
    ),
}


def audit_siren_substrate_readiness(repo_root: Path | str = ".") -> dict[str, Any]:
    """Return a machine-readable, fail-closed local readiness manifest."""

    root = Path(repo_root)
    blockers: list[str] = []
    warnings: list[str] = []
    evidence: dict[str, Any] = {}

    texts: dict[str, str] = {}
    for key, rel in REQUIRED_PATHS.items():
        path = root / rel
        exists = path.is_file()
        evidence[f"{key}_path"] = str(rel)
        evidence[f"{key}_exists"] = exists
        if not exists:
            blockers.append(f"missing_{key}:{rel}")
            continue
        text = path.read_text(encoding="utf-8")
        texts[key] = text
        evidence[f"{key}_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()

    if "trainer" in texts:
        _audit_trainer(texts["trainer"], blockers, evidence)
    if "recipe" in texts:
        _audit_recipe(texts["recipe"], blockers, evidence)
    if "archive" in texts:
        _audit_archive(texts["archive"], blockers, evidence)
    if "inflate" in texts:
        _audit_inflate(texts["inflate"], blockers, warnings, evidence)
    if "score_aware_loss" in texts:
        _audit_score_aware_loss(texts["score_aware_loss"], blockers, evidence)

    local_contract_ready = not blockers
    payload: dict[str, Any] = {
        "schema": "siren_substrate_readiness_v1",
        "lane_id": LANE_ID,
        "summary": (
            "SIREN first-anchor local surfaces are present"
            if local_contract_ready
            else "SIREN first-anchor local surfaces are incomplete"
        ),
        "local_contract_ready": local_contract_ready,
        "ready_for_first_anchor_training": local_contract_ready,
        "ready_for_remote_dispatch": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "score_claim": False,
        "promotion_eligible": False,
        "dispatch_blockers": [
            "operator_authorization_required",
            "active_lane_dispatch_claim_required_before_gpu_spend",
            "no_gpu_spend_in_readiness_gate",
        ],
        "local_blockers": blockers,
        "warnings": warnings,
        "evidence": evidence,
        "operator_commands": {
            "readiness": ".venv/bin/python tools/audit_siren_substrate_readiness.py --json",
            "cpu_smoke": (
                ".venv/bin/python experiments/train_substrate_siren.py "
                "--video-path upstream/videos/0.mkv "
                "--output-dir experiments/results/siren_smoke_<utc> "
                "--epochs 3 --device cpu --smoke --skip-archive-build --skip-auth-eval "
                "# deterministic-bytes acceptable"
            ),
        },
    }
    payload["manifest_hash"] = _manifest_hash(payload)
    return payload


def _audit_trainer(text: str, blockers: list[str], evidence: dict[str, Any]) -> None:
    missing_flags = [flag for flag in REQUIRED_TRAINER_FLAGS if flag not in text]
    evidence["trainer_required_flags_present"] = not missing_flags
    evidence["trainer_missing_required_flags"] = missing_flags
    if missing_flags:
        blockers.append("trainer_missing_required_flags:" + ",".join(missing_flags))

    checks = {
        "uses_canonical_trainer_skeleton": "tac.substrates._shared.trainer_skeleton" in text,
        "declares_tier_1_operator_flags": "TIER_1_OPERATOR_REQUIRED_FLAGS" in text,
        "has_smoke_main": "def _smoke_main" in text and "--smoke" in text,
        "has_full_main": "def _full_main" in text,
        "builds_siren_archive": "from tac.substrates.siren.archive import pack_archive" in text
        or "pack_archive(" in text,
        "emits_runtime": "def _write_runtime" in text and "inflate.sh" in text,
        "builds_archive_zip": "def _build_archive_zip" in text and "archive.zip" in text,
        "runs_auth_eval_path": "CONTEST_AUTH_EVAL_SCRIPT" in text
        and "contest_auth_eval_cuda.json" in text,
        "fails_closed_for_promotion": '"promotion_eligible": False' in text,
        "fails_closed_for_exact_eval_dispatch": '"ready_for_exact_eval_dispatch": False'
        in text,
        "proxy_lagrangian_marked_non_authoritative": "training_proxy_non_authoritative"
        in text
        and '"best_val_lagrangian_score_claim": False' in text
        and '"best_val_lagrangian_promotion_eligible": False' in text,
        "proxy_score_authority_false": '"proxy_score_authority": False' in text,
        "score_claim_requires_exact_auth_eval_score": '"score_claim": contest_cuda_score is not None'
        in text,
        "declares_dispatch_contract": "--dispatch-contract" in text
        and "SIREN_DISPATCH_CONTRACT" in text
        and "require_train_substrate_siren_contract" in text,
    }
    evidence.update(checks)
    for name, passed in checks.items():
        if not passed:
            blockers.append(f"trainer_{name}_missing")


def _audit_recipe(text: str, blockers: list[str], evidence: dict[str, Any]) -> None:
    checks = {
        "lane_id_matches": f"lane_id: {LANE_ID}" in text,
        "remote_driver_declared": "remote_driver:" in text,
        "required_video_declared": "upstream/videos/0.mkv" in text,
        "trainer_declared": "experiments/train_substrate_siren.py" in text,
        "readiness_gate_declared": "tools/audit_siren_substrate_readiness.py" in text,
        "active_contract_is_naked_replacement": (
            "active_dispatch_contract: naked_siren_replacement" in text
            and "SIREN_DISPATCH_CONTRACT: naked_siren_replacement" in text
        ),
    }
    evidence.update({f"recipe_{k}": v for k, v in checks.items()})
    for name, passed in checks.items():
        if not passed:
            blockers.append(f"recipe_{name}_missing")

    modes = _extract_target_modes(text)
    valid_modes = [mode for mode in modes if mode in VALID_TARGET_MODES]
    evidence["recipe_target_modes"] = modes
    evidence["recipe_valid_target_modes"] = valid_modes
    if not valid_modes:
        blockers.append("recipe_target_modes_missing_or_invalid")

    missing_contracts = [
        contract for contract in REQUIRED_DISPATCH_CONTRACTS if contract not in text
    ]
    evidence["recipe_required_dispatch_contracts"] = list(REQUIRED_DISPATCH_CONTRACTS)
    evidence["recipe_missing_dispatch_contracts"] = missing_contracts
    if missing_contracts:
        blockers.append("recipe_missing_dispatch_contracts:" + ",".join(missing_contracts))


def _audit_archive(text: str, blockers: list[str], evidence: dict[str, Any]) -> None:
    checks = {
        "srv1_magic_declared": "SRV1_MAGIC" in text and 'b"SRV1"' in text,
        "monolithic_0bin_declared": "monolithic single-file ``0.bin``" in text
        or "0.bin" in text,
        "pack_archive_declared": "def pack_archive" in text,
        "parse_archive_declared": "def parse_archive" in text,
        "fixed_header_declared": "SRV1_HEADER_FMT" in text and "SRV1_HEADER_SIZE" in text,
    }
    evidence.update({f"archive_{k}": v for k, v in checks.items()})
    for name, passed in checks.items():
        if not passed:
            blockers.append(f"archive_{name}_missing")


def _audit_inflate(
    text: str,
    blockers: list[str],
    warnings: list[str],
    evidence: dict[str, Any],
) -> None:
    checks = {
        "has_inflate_one_video": "def inflate_one_video" in text,
        "has_main_cli": "def main_cli" in text,
        "uses_parser": "parse_archive" in text,
        "loads_siren_model": "SirenSubstrate" in text,
        "no_scorer_imports": "tac.scorer" not in text and "load_differentiable_scorers" not in text,
        "no_pil_dependency": "PIL" not in text and "Image" not in text,
        "writes_contest_raw": ".raw" in text and "write_rgb_pair_to_raw" in text,
        "no_png_writer": "_write_png_rgb" not in text and "zlib.compress" not in text,
    }
    evidence.update({f"inflate_{k}": v for k, v in checks.items()})
    for name, passed in checks.items():
        if not passed:
            blockers.append(f"inflate_{name}_missing")

    loc = len([line for line in text.splitlines() if line.strip()])
    evidence["inflate_nonblank_loc"] = loc
    if loc > 100:
        warnings.append(f"inflate_runtime_loc_budget_exceeded:{loc}")


def _audit_score_aware_loss(
    text: str, blockers: list[str], evidence: dict[str, Any]
) -> None:
    checks = {
        "uses_canonical_score_pair_components": "score_pair_components" in text,
        "uses_shared_contest_constants": "CONTEST_RATE_WEIGHT" in text
        and "CONTEST_SEG_WEIGHT" in text
        and "CONTEST_POSE_SQRT_WEIGHT" in text,
        "forbids_eval_roundtrip_false": "apply_eval_roundtrip=False is forbidden" in text,
        "has_contest_lagrangian_terms": "alpha_rate" in text
        and "beta_seg" in text
        and "gamma_pose" in text,
    }
    evidence.update({f"score_loss_{k}": v for k, v in checks.items()})
    for name, passed in checks.items():
        if not passed:
            blockers.append(f"score_loss_{name}_missing")


def _extract_target_modes(text: str) -> list[str]:
    modes: list[str] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("target_modes:"):
            rest = stripped.split(":", 1)[1].strip()
            if rest.startswith("[") and rest.endswith("]"):
                return [
                    item.strip().strip("'\"")
                    for item in rest[1:-1].split(",")
                    if item.strip()
                ]
            for child in lines[i + 1 :]:
                child_stripped = child.strip()
                if not child.startswith(" ") and not child.startswith("\t"):
                    break
                if child_stripped.startswith("-"):
                    modes.append(child_stripped[1:].strip().strip("'\""))
            return modes
    return modes


def _manifest_hash(payload: dict[str, Any]) -> str:
    clone = {k: v for k, v in payload.items() if k != "manifest_hash"}
    data = json.dumps(clone, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


__all__ = ["LANE_ID", "audit_siren_substrate_readiness"]
