# SPDX-License-Identifier: MIT
"""Tests for BUILD-1 NeRV-trio L0 SCAFFOLD trainers (2026-05-20).

Covers the 3 substrate trainers landed via the BUILD-1 NeRV-trio queue fill:

  - experiments/train_substrate_ego_nerv.py
  - experiments/train_substrate_e_nerv.py
  - experiments/train_substrate_nervdc.py

Each trainer mirrors the canonical tc_nerv skeleton:
  - importable + decorator validates SubstrateContract (Catalog #241/#242)
  - TIER_1_OPERATOR_REQUIRED_FLAGS dict declared (Catalog #151)
  - _smoke_main runs end-to-end on CPU with synthetic batches (Catalog #114)
  - _full_main raises NotImplementedError per Catalog #240 + #315 + #325
  - SubstrateContract declares research_only=true + RESEARCH_ONLY mechanism

These tests are lightweight (CPU-only, ~2-5s each) so they integrate with
the canonical pytest collection.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINERS = {
    "ego_nerv": REPO_ROOT / "experiments" / "train_substrate_ego_nerv.py",
    "e_nerv": REPO_ROOT / "experiments" / "train_substrate_e_nerv.py",
    "nervdc": REPO_ROOT / "experiments" / "train_substrate_nervdc.py",
}


_TRAINER_MODULE_CACHE: dict[str, object] = {}


def _import_trainer(name: str):
    """Import a trainer module from its file path; return the module.

    Cached at process scope because the @register_substrate decorator is
    non-reentrant (raises SubstrateContractError on duplicate id when the
    module is re-imported). Cache hit returns the same module instance, which
    is sufficient for all assertions in this test file (they read attributes,
    not re-register the contract).
    """
    if name in _TRAINER_MODULE_CACHE:
        return _TRAINER_MODULE_CACHE[name]
    path = TRAINERS[name]
    assert path.is_file(), f"trainer not found: {path}"
    spec = importlib.util.spec_from_file_location(f"train_substrate_{name}", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _TRAINER_MODULE_CACHE[name] = module
    return module


@pytest.mark.parametrize("substrate", list(TRAINERS.keys()))
def test_trainer_importable(substrate):
    """Trainer module imports cleanly, exposing main() + SubstrateContract."""
    m = _import_trainer(substrate)
    assert hasattr(m, "main"), f"{substrate}: missing main()"
    assert hasattr(m, f"{substrate.upper()}_SUBSTRATE_CONTRACT"), (
        f"{substrate}: missing SubstrateContract"
    )


@pytest.mark.parametrize("substrate", list(TRAINERS.keys()))
def test_trainer_declares_tier_1_manifest(substrate):
    """Catalog #151 + #168: TIER_1_OPERATOR_REQUIRED_FLAGS dict declared."""
    m = _import_trainer(substrate)
    assert hasattr(m, "TIER_1_OPERATOR_REQUIRED_FLAGS"), (
        f"{substrate}: missing TIER_1_OPERATOR_REQUIRED_FLAGS"
    )
    manifest = m.TIER_1_OPERATOR_REQUIRED_FLAGS
    assert isinstance(manifest, dict)
    # Required flags per the canonical tc_nerv pattern.
    for required_flag in ("--video-path", "--output-dir", "--epochs",
                           "--upstream-dir", "--device"):
        assert required_flag in manifest, (
            f"{substrate}: missing required flag {required_flag} in TIER_1 manifest"
        )
        entry = manifest[required_flag]
        assert "env" in entry, f"{substrate}: {required_flag} missing 'env'"
        assert "rationale" in entry, f"{substrate}: {required_flag} missing 'rationale'"


@pytest.mark.parametrize("substrate", list(TRAINERS.keys()))
def test_substrate_contract_is_research_only(substrate):
    """SubstrateContract declares research_only=true (L0 SCAFFOLD posture)."""
    m = _import_trainer(substrate)
    contract = getattr(m, f"{substrate.upper()}_SUBSTRATE_CONTRACT")
    assert contract.recipe_research_only is True, (
        f"{substrate}: SubstrateContract should declare recipe_research_only=True "
        f"at L0 SCAFFOLD posture per CLAUDE.md 'Substrate scaffolds MUST be "
        f"COMPLETE or RESEARCH-ONLY' non-negotiable"
    )
    assert contract.score_improvement_mechanism_status == "RESEARCH_ONLY", (
        f"{substrate}: SubstrateContract should declare "
        f"score_improvement_mechanism_status=RESEARCH_ONLY at L0 SCAFFOLD posture"
    )


@pytest.mark.parametrize("substrate", list(TRAINERS.keys()))
def test_substrate_contract_target_modes_research_substrate(substrate):
    """SubstrateContract target_modes contains research_substrate."""
    m = _import_trainer(substrate)
    contract = getattr(m, f"{substrate.upper()}_SUBSTRATE_CONTRACT")
    assert "research_substrate" in contract.target_modes, (
        f"{substrate}: target_modes should contain research_substrate at "
        f"L0 SCAFFOLD posture; got {contract.target_modes}"
    )


@pytest.mark.parametrize("substrate", list(TRAINERS.keys()))
def test_smoke_runs_end_to_end(tmp_path, substrate):
    """_smoke_main runs CPU-only smoke end-to-end and writes provenance.

    Verifies (i) the trainer's argparse + dispatch path works, (ii) the
    substrate library at src/tac/<name>_as_renderer.py is importable +
    forward-pass-functional, (iii) the provenance.json schema is honest
    about score_claim=false + promotion_eligible=false (Catalog #127/#192/#323).
    """
    out_dir = tmp_path / f"{substrate}_smoke"
    trainer_path = TRAINERS[substrate]
    proc = subprocess.run(
        [
            sys.executable,
            str(trainer_path),
            "--output-dir", str(out_dir),
            "--epochs", "1",
            "--device", "cpu",
            "--smoke",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"{substrate}: smoke failed rc={proc.returncode}\n"
        f"stdout: {proc.stdout[-500:]}\nstderr: {proc.stderr[-500:]}"
    )
    smoke_ckpt = out_dir / "smoke_checkpoint.pt"
    assert smoke_ckpt.is_file(), f"{substrate}: smoke_checkpoint.pt not written"
    provenance = out_dir / "provenance.json"
    assert provenance.is_file(), f"{substrate}: provenance.json not written"
    import json
    p = json.loads(provenance.read_text())
    assert p["smoke"] is True
    assert p["score_claim"] is False
    assert p["promotion_eligible"] is False
    assert p["ready_for_exact_eval_dispatch"] is False


@pytest.mark.parametrize("substrate", list(TRAINERS.keys()))
def test_full_main_raises_not_implemented(tmp_path, substrate):
    """_full_main raises NotImplementedError per Catalog #240 + #315 + #325."""
    out_dir = tmp_path / f"{substrate}_full"
    trainer_path = TRAINERS[substrate]
    proc = subprocess.run(
        [
            sys.executable,
            str(trainer_path),
            "--output-dir", str(out_dir),
            "--epochs", "1",
            "--device", "cpu",
            # NOTE: --smoke omitted so _full_main fires
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0, (
        f"{substrate}: _full_main should have raised NotImplementedError "
        f"but exited rc=0"
    )
    # Verify the error message cites the canonical catalog gates.
    combined = (proc.stdout + proc.stderr).lower()
    assert "notimplementederror" in combined, (
        f"{substrate}: NotImplementedError not in output:\n"
        f"stdout: {proc.stdout[-200:]}\nstderr: {proc.stderr[-500:]}"
    )
    assert "catalog #240" in combined or "catalog #315" in combined or "catalog #325" in combined, (
        f"{substrate}: error message should cite Catalog #240/#315/#325; got:\n"
        f"{proc.stderr[-500:]}"
    )


@pytest.mark.parametrize("substrate", list(TRAINERS.keys()))
def test_lane_registry_entry_exists(substrate):
    """Lane registry entry exists at lane_<substrate>_l0_scaffold_20260520."""
    import json
    registry_path = REPO_ROOT / ".omx" / "state" / "lane_registry.json"
    assert registry_path.is_file(), "lane_registry.json missing"
    registry = json.loads(registry_path.read_text())
    expected_lane_id = f"lane_{substrate}_l0_scaffold_20260520"
    lane_ids = {lane["id"] for lane in registry["lanes"]}
    assert expected_lane_id in lane_ids, (
        f"{substrate}: lane {expected_lane_id} not registered; "
        f"see tools/lane_maturity.py add-lane"
    )


@pytest.mark.parametrize("substrate", list(TRAINERS.keys()))
def test_recipe_exists_and_is_research_only(substrate):
    """L0 SCAFFOLD recipe exists + declares research_only:true + dispatch_enabled:false."""
    recipe_path = (
        REPO_ROOT / ".omx" / "operator_authorize_recipes" /
        f"substrate_{substrate}_modal_a10g_diagnostic_dispatch.yaml"
    )
    assert recipe_path.is_file(), f"{substrate}: scaffold recipe missing at {recipe_path}"
    text = recipe_path.read_text()
    assert "research_only: true" in text, (
        f"{substrate}: recipe must declare research_only: true"
    )
    assert "dispatch_enabled: false" in text, (
        f"{substrate}: recipe must declare dispatch_enabled: false"
    )


@pytest.mark.parametrize("substrate", list(TRAINERS.keys()))
def test_driver_exists_and_passes_bash_syntax(substrate):
    """L0 SCAFFOLD remote driver exists + passes bash -n syntax check."""
    driver_path = (
        REPO_ROOT / "scripts" / f"remote_lane_substrate_{substrate}_l0_scaffold.sh"
    )
    assert driver_path.is_file(), f"{substrate}: scaffold driver missing at {driver_path}"
    proc = subprocess.run(
        ["bash", "-n", str(driver_path)],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, (
        f"{substrate}: bash -n failed for driver:\n{proc.stderr}"
    )
    # Catalog #244: canonical NVML env block must be present.
    text = driver_path.read_text()
    assert "DALI_DISABLE_NVML" in text, (
        f"{substrate}: driver missing DALI_DISABLE_NVML per Catalog #244"
    )
    assert "CUBLAS_WORKSPACE_CONFIG" in text, (
        f"{substrate}: driver missing CUBLAS_WORKSPACE_CONFIG per Catalog #244"
    )
    assert "PYTORCH_CUDA_ALLOC_CONF" in text, (
        f"{substrate}: driver missing PYTORCH_CUDA_ALLOC_CONF per Catalog #244"
    )


@pytest.mark.parametrize("substrate", list(TRAINERS.keys()))
def test_design_memo_exists_with_required_sections(substrate):
    """Design memo exists + carries the 5 binding section headers per
    Catalog #290 (canonical-vs-unique) + #294 (9-dim checklist) +
    #303 (cargo-cult audit) + #305 (observability) + #309 (horizon-class)."""
    memo_path = (
        REPO_ROOT / ".omx" / "research" /
        f"{substrate}_l0_scaffold_design_20260520T140000Z.md"
    )
    assert memo_path.is_file(), f"{substrate}: design memo missing at {memo_path}"
    text = memo_path.read_text()
    # Catalog #290: canonical-vs-unique decision per layer
    assert "## Canonical-vs-unique decision per layer" in text, (
        f"{substrate}: design memo missing Catalog #290 section header"
    )
    # Catalog #294: 9-dimension success checklist evidence
    assert "## 9-dimension success checklist evidence" in text, (
        f"{substrate}: design memo missing Catalog #294 section header"
    )
    # Catalog #303: cargo-cult audit per assumption
    assert "## Cargo-cult audit per assumption" in text, (
        f"{substrate}: design memo missing Catalog #303 section header"
    )
    # Catalog #305: observability surface
    assert "## Observability surface" in text, (
        f"{substrate}: design memo missing Catalog #305 section header"
    )
    # Catalog #309: horizon-class declaration in frontmatter
    assert "horizon_class:" in text, (
        f"{substrate}: design memo missing Catalog #309 horizon_class declaration"
    )
