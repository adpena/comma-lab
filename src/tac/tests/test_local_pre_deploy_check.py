# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools/local_pre_deploy_check.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("local_pre_deploy_check", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_auth_eval_reachability_ignores_artifact_globs(tmp_path: Path) -> None:
    module = _load_module()
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "from pathlib import Path\n"
        "def main():\n"
        "    return list(Path('x').glob('contest_auth_eval*.json'))\n",
        encoding="utf-8",
    )

    passed, message = module.check_auth_eval_reachability(trainer)

    assert passed is False
    assert "no reachable auth_eval invocation" in message


def test_auth_eval_reachability_accepts_canonical_helper_call(tmp_path: Path) -> None:
    module = _load_module()
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "def main():\n"
        "    gate_auth_eval_call(archive='archive.zip')\n",
        encoding="utf-8",
    )

    passed, message = module.check_auth_eval_reachability(trainer)

    assert passed is True
    assert "reachable auth_eval invocation" in message


def test_auth_eval_reachability_rejects_unreachable_helper_call(
    tmp_path: Path,
) -> None:
    module = _load_module()
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "def unused_auth_eval():\n"
        "    gate_auth_eval_call(archive='archive.zip')\n"
        "\n"
        "def main():\n"
        "    return 0\n",
        encoding="utf-8",
    )

    passed, message = module.check_auth_eval_reachability(trainer)

    assert passed is False
    assert "no reachable auth_eval invocation" in message


def test_auth_eval_reachability_rejects_dead_contest_auth_eval_command_literal(
    tmp_path: Path,
) -> None:
    module = _load_module()
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "def main():\n"
        "    cmd = ['python', 'experiments/contest_auth_eval.py']\n"
        "    return cmd\n",
        encoding="utf-8",
    )

    passed, message = module.check_auth_eval_reachability(trainer)

    assert passed is False
    assert "no reachable auth_eval invocation" in message


def test_auth_eval_reachability_rejects_local_run_wrapper_command_literal(
    tmp_path: Path,
) -> None:
    module = _load_module()
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "def run(cmd):\n"
        "    return cmd\n"
        "\n"
        "def main():\n"
        "    return run(['python', 'experiments/contest_auth_eval.py'])\n",
        encoding="utf-8",
    )

    passed, message = module.check_auth_eval_reachability(trainer)

    assert passed is False
    assert "no reachable auth_eval invocation" in message


def test_auth_eval_reachability_accepts_reachable_helper_wrapper(
    tmp_path: Path,
) -> None:
    module = _load_module()
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "def run_auth_eval():\n"
        "    return gate_auth_eval_call(archive='archive.zip')\n"
        "\n"
        "def _full_main(args):\n"
        "    return run_auth_eval()\n",
        encoding="utf-8",
    )

    passed, message = module.check_auth_eval_reachability(trainer)

    assert passed is True
    assert "reachable" in message


def test_auth_eval_reachability_accepts_reachable_subprocess_invocation(
    tmp_path: Path,
) -> None:
    module = _load_module()
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "import subprocess\n"
        "\n"
        "def main():\n"
        "    return subprocess.run(['python', 'experiments/contest_auth_eval.py'], check=True)\n",
        encoding="utf-8",
    )

    passed, message = module.check_auth_eval_reachability(trainer)

    assert passed is True
    assert "reachable" in message


def test_trainer_importable_registers_dataclass_module(tmp_path: Path) -> None:
    module = _load_module()
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class TrainerConfig:\n"
        "    width: int = 8\n",
        encoding="utf-8",
    )

    passed, message = module.check_trainer_importable(trainer)

    assert passed is True
    assert "imports cleanly" in message


def test_archive_grammar_reuses_imported_substrate_contract_module(
    tmp_path: Path,
) -> None:
    module = _load_module()
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "from pathlib import Path\n"
        "import zipfile\n"
        "from tac.substrate_registry import SubstrateContract, register_substrate\n"
        "from tac.substrate_registry.contract import NOT_APPLICABLE_WITH_RATIONALE\n"
        "\n"
        "CONTRACT = SubstrateContract(\n"
        "    id='local_predeploy_reimport',\n"
        "    lane_id='lane_local_predeploy_reimport_20260515',\n"
        "    target_modes=('research_substrate',),\n"
        "    deployment_target='desktop_research',\n"
        "    council_verdict_provenance=None,\n"
        "    archive_grammar='single 0.bin',\n"
        "    parser_section_manifest={'header': 'magic'},\n"
        "    inflate_runtime_loc_budget=80,\n"
        "    runtime_dep_closure=('torch',),\n"
        "    export_format='fp16_brotli',\n"
        "    score_aware_loss='scorer_loss_terms_btchw',\n"
        "    bolt_on_loc_budget=200,\n"
        "    no_op_detector_planned=True,\n"
        "    archive_bytes_added=None,\n"
        "    score_improvement_mechanism_status='RESEARCH_ONLY',\n"
        "    runtime_overlay_consumed=False,\n"
        "    recipe_smoke_only=True,\n"
        "    recipe_research_only=True,\n"
        "    recipe_min_smoke_gpu='T4',\n"
        "    recipe_min_vram_gb=16,\n"
        "    recipe_pyav_decode_strategy='cpu_thread_async_upload',\n"
        "    recipe_canary_status='independent_substrate',\n"
        "    recipe_video_input_strategy='per_dispatch_local_copy',\n"
        "    recipe_canary_dependency=None,\n"
        "    cost_band_epochs=10,\n"
        "    cost_band_gpu_key='T4',\n"
        "    cost_band_platform_key='modal',\n"
        "    cost_band_p50_usd=0.10,\n"
        "    hook_sensitivity_contribution=NOT_APPLICABLE_WITH_RATIONALE,\n"
        "    hook_pareto_constraint=NOT_APPLICABLE_WITH_RATIONALE,\n"
        "    hook_bit_allocator_class=NOT_APPLICABLE_WITH_RATIONALE,\n"
        "    hook_autopilot_ranker_class_shift_token=None,\n"
        "    hook_continual_learning_anchor_kind=NOT_APPLICABLE_WITH_RATIONALE,\n"
        "    hook_probe_disambiguator=None,\n"
        "    catalog_compliance_declarations=('catalog_205_select_inflate_device_used',),\n"
        "    hook_not_applicable_rationale={\n"
        "        'hook_sensitivity_contribution': 'test',\n"
        "        'hook_pareto_constraint': 'test',\n"
        "        'hook_bit_allocator_class': 'test',\n"
        "        'hook_continual_learning_anchor_kind': 'test',\n"
        "        'hook_probe_disambiguator': 'test',\n"
        "    },\n"
        ")\n"
        "\n"
        "@register_substrate(CONTRACT)\n"
        "def main():\n"
        "    return 0\n"
        "\n"
        "def _build_archive_zip(archive_zip_path: Path, *, bin_bytes: bytes) -> None:\n"
        "    with zipfile.ZipFile(archive_zip_path, 'w') as zf:\n"
        "        zi = zipfile.ZipInfo('0.bin', date_time=(1980, 1, 1, 0, 0, 0))\n"
        "        zf.writestr(zi, bin_bytes)\n",
        encoding="utf-8",
    )

    import_passed, import_message = module.check_trainer_importable(trainer)
    grammar_passed, grammar_message = module.check_archive_grammar(trainer)

    assert import_passed is True, import_message
    assert grammar_passed is True, grammar_message


def test_archive_grammar_skips_dynamic_check_when_builder_needs_runtime_dir(
    tmp_path: Path,
) -> None:
    module = _load_module()
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "from pathlib import Path\n"
        "import zipfile\n"
        "\n"
        "def _build_archive_zip(\n"
        "    archive_zip_path: Path,\n"
        "    *,\n"
        "    bin_bytes: bytes,\n"
        "    submission_dir: Path,\n"
        ") -> None:\n"
        "    with zipfile.ZipFile(archive_zip_path, 'w') as zf:\n"
        "        zi = zipfile.ZipInfo('0.bin', date_time=(1980, 1, 1, 0, 0, 0))\n"
        "        zf.writestr(zi, bin_bytes)\n"
        "        (submission_dir / 'inflate.sh').read_text()\n",
        encoding="utf-8",
    )

    passed, message = module.check_archive_grammar(trainer)

    assert passed is True
    assert "dynamic check skipped" in message
    assert "submission_dir" in message


def test_archive_grammar_rejects_x_without_runtime_evidence(tmp_path: Path) -> None:
    module = _load_module()
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "from pathlib import Path\n"
        "import zipfile\n"
        "\n"
        "def _build_archive_zip(archive_zip_path: Path, *, bin_bytes: bytes) -> None:\n"
        "    with zipfile.ZipFile(archive_zip_path, 'w') as zf:\n"
        "        zi = zipfile.ZipInfo('x', date_time=(1980, 1, 1, 0, 0, 0))\n"
        "        zf.writestr(zi, bin_bytes)\n",
        encoding="utf-8",
    )

    passed, message = module.check_archive_grammar(trainer)

    assert passed is False
    assert "member 'x' requires explicit runtime evidence" in message


def test_archive_grammar_accepts_x_with_runtime_evidence(tmp_path: Path) -> None:
    module = _load_module()
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "from pathlib import Path\n"
        "import zipfile\n"
        "\n"
        "def _write_runtime() -> str:\n"
        "    return 'SRC=\"${DATA_DIR}/x\"\\n'\n"
        "\n"
        "def _build_archive_zip(archive_zip_path: Path, *, bin_bytes: bytes) -> None:\n"
        "    with zipfile.ZipFile(archive_zip_path, 'w') as zf:\n"
        "        zi = zipfile.ZipInfo('x', date_time=(1980, 1, 1, 0, 0, 0))\n"
        "        zf.writestr(zi, bin_bytes)\n",
        encoding="utf-8",
    )

    passed, message = module.check_archive_grammar(trainer)

    assert passed is True
    assert "archive ZIP member" in message


def test_archive_grammar_accepts_x_with_explicit_grammar_evidence(
    tmp_path: Path,
) -> None:
    module = _load_module()
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "from pathlib import Path\n"
        "import zipfile\n"
        "\n"
        "archive_grammar = 'a1 single_member:x'\n"
        "\n"
        "def _build_archive_zip(archive_zip_path: Path, *, bin_bytes: bytes) -> None:\n"
        "    with zipfile.ZipFile(archive_zip_path, 'w') as zf:\n"
        "        zi = zipfile.ZipInfo('x', date_time=(1980, 1, 1, 0, 0, 0))\n"
        "        zf.writestr(zi, bin_bytes)\n",
        encoding="utf-8",
    )

    passed, message = module.check_archive_grammar(trainer)

    assert passed is True
    assert "archive ZIP member" in message


def test_archive_grammar_accepts_x_with_same_line_waiver(tmp_path: Path) -> None:
    module = _load_module()
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "from pathlib import Path\n"
        "import zipfile\n"
        "\n"
        "def _build_archive_zip(archive_zip_path: Path, *, bin_bytes: bytes) -> None:\n"
        "    with zipfile.ZipFile(archive_zip_path, 'w') as zf:\n"
        "        zi = zipfile.ZipInfo('x', date_time=(1980, 1, 1, 0, 0, 0))  # ARCHIVE_MEMBER_OK:intentional-x-runtime\n"
        "        zf.writestr(zi, bin_bytes)\n",
        encoding="utf-8",
    )

    passed, message = module.check_archive_grammar(trainer)

    assert passed is True
    assert "archive ZIP member" in message


def _write_recipe(tmp_path: Path, name: str, text: str) -> None:
    recipes = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes.mkdir(parents=True)
    (recipes / f"{name}.yaml").write_text(text, encoding="utf-8")


def test_recipe_state_rejects_implemented_trainer_with_dispatch_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-deploy must match operator_authorize refusal for disabled recipes."""
    module = _load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "def _full_main(args):\n"
        "    return 0\n",
        encoding="utf-8",
    )
    _write_recipe(
        tmp_path,
        "disabled_recipe",
        "schema_version: 1\n"
        "name: disabled_recipe\n"
        "dispatch_enabled: false\n"
        "dispatch_blockers:\n"
        "  - phase_2_gate\n",
    )

    passed, message = module.check_recipe_status_consistent_with_trainer_state(
        trainer, "disabled_recipe"
    )

    assert passed is False
    assert "non-dispatchable" in message
    assert "dispatch_enabled=false" in message
    assert "dispatch_blockers" in message
    assert "operator_authorize.py would refuse" in message


def test_recipe_state_accepts_implemented_dispatchable_recipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "def _full_main(args):\n"
        "    return 0\n",
        encoding="utf-8",
    )
    _write_recipe(
        tmp_path,
        "enabled_recipe",
        "schema_version: 1\n"
        "name: enabled_recipe\n"
        "research_only: false\n"
        "dispatch_enabled: true\n",
    )

    passed, message = module.check_recipe_status_consistent_with_trainer_state(
        trainer, "enabled_recipe"
    )

    assert passed is True
    assert "recipe is dispatchable" in message


def test_recipe_state_accepts_implemented_research_only_dispatchable_recipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "def _full_main(args):\n"
        "    return 0\n",
        encoding="utf-8",
    )
    _write_recipe(
        tmp_path,
        "research_smoke_recipe",
        "schema_version: 1\n"
        "name: research_smoke_recipe\n"
        "research_only: true\n"
        "dispatch_enabled: true\n"
        "score_claim: false\n"
        "promotion_eligible: false\n",
    )

    passed, message = module.check_recipe_status_consistent_with_trainer_state(
        trainer, "research_smoke_recipe"
    )

    assert passed is True
    assert "dispatchable" in message
    assert "research_only=true false-authority metadata preserved" in message


def test_recipe_state_accepts_notimplemented_when_recipe_non_dispatchable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "def _full_main(args):\n"
        "    raise NotImplementedError('phase gate')\n",
        encoding="utf-8",
    )
    _write_recipe(
        tmp_path,
        "research_recipe",
        "schema_version: 1\n"
        "name: research_recipe\n"
        "research_only: true\n"
        "dispatch_enabled: false\n",
    )

    passed, message = module.check_recipe_status_consistent_with_trainer_state(
        trainer, "research_recipe"
    )

    assert passed is True
    assert "transparent non-dispatchable" in message


def test_recipe_state_rejects_notimplemented_when_only_research_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "def _full_main(args):\n"
        "    raise NotImplementedError('phase gate')\n",
        encoding="utf-8",
    )
    _write_recipe(
        tmp_path,
        "research_metadata_only_recipe",
        "schema_version: 1\n"
        "name: research_metadata_only_recipe\n"
        "research_only: true\n",
    )

    passed, message = module.check_recipe_status_consistent_with_trainer_state(
        trainer, "research_metadata_only_recipe"
    )

    assert passed is False
    assert "not a dispatch refusal" in message
