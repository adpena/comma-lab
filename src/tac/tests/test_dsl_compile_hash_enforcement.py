"""Catalog #406: DSL compile hash launcher/governor fail-closed coverage."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tac import admission_guard as admission
from tac.preflight import check_launch_and_governor_require_dsl_compile_hash
from tac.v9_provenance_gates import (
    DSL_COMPILE_HASH_ENV,
    DSL_LAUNCH_MANIFEST_SCHEMA,
    DSL_LAUNCH_SH_PATH_ENV,
    DSL_PROVENANCE_PATH_ENV,
    DSL_PROVENANCE_SCHEMA,
    _hash_payload,
    build_dsl_compile_provenance_document,
    canonicalize_resolved_argv,
    verify_dsl_provenance_artifacts,
    verify_dsl_provenance_document,
)
from tac.witness_dsl.curriculum_dsl import TRAINER_REL

REPO = Path(__file__).resolve().parents[3]


def _load_tool(name: str):
    path = REPO / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _document(*, seed: int = 7) -> dict:
    doc = {
        "schema": DSL_PROVENANCE_SCHEMA,
        "spec_id": "test_program",
        "witness_program_spec": {"name": "test_program", "seed": seed},
        "resolved_argv": list(
            canonicalize_resolved_argv(
                [".venv/bin/python", TRAINER_REL, "--out-dir", "/volatile/run", "--seed", seed]
            )
        ),
        "bijection_manifest": {
            "program": "test_program",
            "bindings": [{"flag": "--seed", "lever_owners": ["SeedLever"]}],
        },
        "lawref_provenance": {"--seed": {"equation_id": "seed_identity_v1"}},
        "non_authoritative_context": {"compiled_at_utc": "changes freely"},
    }
    doc["bijection_hash"] = _hash_payload(doc["bijection_manifest"])
    payload = {
        "schema": DSL_PROVENANCE_SCHEMA,
        "spec_id": doc["spec_id"],
        "witness_program_spec": doc["witness_program_spec"],
        "resolved_argv": doc["resolved_argv"],
        "bijection_manifest": doc["bijection_manifest"],
        "bijection_hash": doc["bijection_hash"],
        "lawref_provenance": doc["lawref_provenance"],
    }
    doc["dsl_compile_hash"] = _hash_payload(payload)
    return doc


def _rehash_document(document: dict) -> None:
    payload = {
        "schema": DSL_PROVENANCE_SCHEMA,
        "spec_id": document["spec_id"],
        "witness_program_spec": document["witness_program_spec"],
        "resolved_argv": document["resolved_argv"],
        "bijection_manifest": document["bijection_manifest"],
        "bijection_hash": document["bijection_hash"],
        "lawref_provenance": document["lawref_provenance"],
    }
    document["dsl_compile_hash"] = _hash_payload(payload)


def _write_artifacts(tmp_path: Path, doc: dict | None = None) -> tuple[Path, dict]:
    document = copy.deepcopy(doc or _document())
    launch_sh = tmp_path / "launch.sh"
    provenance = tmp_path / "dsl_provenance.json"
    manifest = tmp_path / "launch_manifest.json"
    exact_tokens = [str(token) for token in document["resolved_argv"]]
    exact_tokens[0] = ".venv/bin/python"
    exact_tokens = [str(tmp_path) if token == "<OUT_DIR>" else token for token in exact_tokens]
    import shlex

    launch_sh.write_text(
        "#!/usr/bin/env bash\n"
        f"# dsl_compile_hash: {document['dsl_compile_hash']}\n"
        f"{shlex.join(exact_tokens)}\n",
        encoding="utf-8",
    )
    provenance.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    launch_bytes = launch_sh.read_bytes()
    provenance_bytes = provenance.read_bytes()
    manifest.write_text(
        json.dumps(
            {
                "schema": DSL_LAUNCH_MANIFEST_SCHEMA,
                "dsl_compile_hash": document["dsl_compile_hash"],
                "launch_sh_sha256": hashlib.sha256(launch_bytes).hexdigest(),
                "dsl_provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return launch_sh, document


@pytest.fixture(scope="session")
def v9_document() -> dict:
    from tac.witness_dsl.spec_v9_cgauge import compile_v9_cgauge_ideal_launch_config

    cfg = compile_v9_cgauge_ideal_launch_config(
        "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
        num_pairs=600,
        mod_dim=19,
        program_name="v9_cgauge_ideal_mod19",
    )
    return build_dsl_compile_provenance_document(
        program_name=cfg.typed.name,
        typed_config=cfg.typed,
        compiler_manifest=cfg.constants_manifest,
        repo_root=REPO,
    )


def _process_argv(document: dict, out_dir: Path) -> list[str]:
    tokens = [str(out_dir) if token == "<OUT_DIR>" else str(token) for token in document["resolved_argv"]]
    return tokens[1:]


def _binding_env(launch_sh: Path, document: dict) -> dict[str, str]:
    return {
        admission.GOVERNED_MARKER_ENV: "1",
        DSL_COMPILE_HASH_ENV: document["dsl_compile_hash"],
        DSL_PROVENANCE_PATH_ENV: str(launch_sh.with_name("dsl_provenance.json")),
        DSL_LAUNCH_SH_PATH_ENV: str(launch_sh),
    }


def test_canonical_argv_excludes_run_identity() -> None:
    a = canonicalize_resolved_argv(["py", TRAINER_REL, "--out-dir", "a", "--seed", 7])
    b = canonicalize_resolved_argv(["py", TRAINER_REL, "--out-dir", "b", "--seed", 7])
    assert a == b


def test_canonical_argv_preserves_interpreter_change() -> None:
    a = canonicalize_resolved_argv(["py-a", TRAINER_REL, "--seed", 7])
    b = canonicalize_resolved_argv(["py-b", TRAINER_REL, "--seed", 7])
    assert a != b


def test_canonical_argv_preserves_semantic_change() -> None:
    a = canonicalize_resolved_argv(["py", TRAINER_REL, "--seed", 7])
    b = canonicalize_resolved_argv(["py", TRAINER_REL, "--seed", 8])
    assert a != b


def test_valid_document_recomputes() -> None:
    doc = _document()
    assert verify_dsl_provenance_document(doc, expected_hash=doc["dsl_compile_hash"])[0]


def test_document_hash_mismatch_refused() -> None:
    doc = _document()
    doc["dsl_compile_hash"] = "0" * 64
    ok, detail = verify_dsl_provenance_document(doc)
    assert not ok and "dsl_compile_hash mismatch" in detail


def test_bijection_hash_mismatch_refused() -> None:
    doc = _document()
    doc["bijection_manifest"]["bindings"][0]["lever_owners"] = ["HandRule"]
    ok, detail = verify_dsl_provenance_document(doc)
    assert not ok and "#332 bijection manifest hash mismatch" in detail


def test_carried_hash_mismatch_refused() -> None:
    doc = _document()
    ok, detail = verify_dsl_provenance_document(doc, expected_hash="f" * 64)
    assert not ok and "carried dsl_compile_hash" in detail


def test_post_compile_argv_edit_refused() -> None:
    doc = _document()
    argv = ["py", TRAINER_REL, "--out-dir", "new", "--seed", "8"]
    ok, detail = verify_dsl_provenance_document(doc, launch_argv=argv)
    assert not ok and "does not round-trip" in detail


def test_valid_artifact_triple_admitted(tmp_path: Path, v9_document: dict) -> None:
    launch_sh, doc = _write_artifacts(tmp_path, v9_document)
    assert verify_dsl_provenance_artifacts(
        launch_sh, expected_hash=doc["dsl_compile_hash"]
    )[0]


@pytest.mark.parametrize("missing", ["dsl_provenance.json", "launch_manifest.json"])
def test_missing_binding_artifact_refused(tmp_path: Path, missing: str) -> None:
    launch_sh, _ = _write_artifacts(tmp_path)
    (tmp_path / missing).unlink()
    ok, detail = verify_dsl_provenance_artifacts(launch_sh)
    assert not ok and "missing" in detail.lower()


def test_missing_launch_header_refused(tmp_path: Path) -> None:
    launch_sh, _ = _write_artifacts(tmp_path)
    launch_sh.write_text(launch_sh.read_text().replace("# dsl_compile_hash: ", "# removed: "))
    ok, detail = verify_dsl_provenance_artifacts(launch_sh)
    assert not ok and "exactly one" in detail


def test_launch_sh_byte_edit_refused(tmp_path: Path) -> None:
    launch_sh, _ = _write_artifacts(tmp_path)
    launch_sh.write_text(launch_sh.read_text().replace("--seed 7", "--seed 8"))
    ok, detail = verify_dsl_provenance_artifacts(launch_sh)
    assert not ok and ("sha256" in detail or "round-trip" in detail)


def test_launch_manifest_hash_edit_refused(tmp_path: Path) -> None:
    launch_sh, _ = _write_artifacts(tmp_path)
    path = tmp_path / "launch_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["launch_sh_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    ok, detail = verify_dsl_provenance_artifacts(launch_sh)
    assert not ok and "launch_sh_sha256" in detail


def test_self_rehashed_lawref_forgery_refused_by_dsl_recompile(
    tmp_path: Path, v9_document: dict
) -> None:
    forged = copy.deepcopy(v9_document)
    lawref = next(iter(forged["lawref_provenance"].values()))
    lawref["equation_id"] = "hand_ruled_equation_v1"
    _rehash_document(forged)
    launch_sh, _ = _write_artifacts(tmp_path, forged)
    ok, detail = verify_dsl_provenance_artifacts(launch_sh)
    assert not ok and "unregistered equation" in detail


def test_governor_refuses_hand_authored_witness_argv() -> None:
    daemon = _load_tool("spawn_durable_daemon")
    args = SimpleNamespace()
    rc = daemon._witness_dsl_compile_hash_gate(
        args, ["python", TRAINER_REL, "--seed", "7"]
    )
    assert rc == 8


def test_governor_refuses_witness_hidden_in_renamed_shell_script(tmp_path: Path) -> None:
    daemon = _load_tool("spawn_durable_daemon")
    raw_script = tmp_path / "looks_innocent.sh"
    raw_script.write_text(
        f"#!/usr/bin/env bash\npython {TRAINER_REL} --seed 7\n",
        encoding="utf-8",
    )
    rc = daemon._witness_dsl_compile_hash_gate(
        SimpleNamespace(), ["bash", str(raw_script)]
    )
    assert rc == 8


def test_governor_admits_recomputed_dsl_artifacts(tmp_path: Path, v9_document: dict) -> None:
    daemon = _load_tool("spawn_durable_daemon")
    launch_sh, doc = _write_artifacts(tmp_path, v9_document)
    args = SimpleNamespace()
    assert daemon._witness_dsl_compile_hash_gate(args, ["bash", str(launch_sh)]) is None
    assert args.dsl_compile_hash == doc["dsl_compile_hash"]


def test_trainer_guard_refuses_marker_without_dsl_binding() -> None:
    ok, detail = admission.admission_status(
        "train_levelset_witness_realized_through_R_mlx",
        env={admission.GOVERNED_MARKER_ENV: "1"},
    )
    assert not ok and "DSL COMPILE REFUSED" in detail


def test_trainer_guard_admits_matching_running_argv(tmp_path: Path, v9_document: dict) -> None:
    launch_sh, doc = _write_artifacts(tmp_path, v9_document)
    ok, detail = admission.admission_status(
        "train_levelset_witness_realized_through_R_mlx",
        env=_binding_env(launch_sh, doc),
        process_argv=_process_argv(doc, tmp_path),
    )
    assert ok, detail


def test_trainer_guard_refuses_different_running_argv(tmp_path: Path, v9_document: dict) -> None:
    launch_sh, doc = _write_artifacts(tmp_path, v9_document)
    running = _process_argv(doc, tmp_path)
    seed_index = running.index("--seed") + 1
    running[seed_index] = str(int(running[seed_index]) + 1)
    ok, detail = admission.admission_status(
        "train_levelset_witness_realized_through_R_mlx",
        env=_binding_env(launch_sh, doc),
        process_argv=running,
    )
    assert not ok and "running argv mismatch" in detail


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dry_run": True},
        {"skip": True},
        {"allow_rationale": "reviewed but retired bypass"},
    ],
)
def test_legacy_launcher_options_cannot_downgrade_refusal(kwargs: dict) -> None:
    launcher = _load_tool("launch_witness_run")
    base = {
        "ok": False,
        "detail": "missing hash",
        "manifest_absent": True,
        "config": "hand_config",
        "dry_run": False,
        "skip": False,
        "enforce": False,
        "allow_rationale": None,
    }
    base.update(kwargs)
    action, message = launcher.dsl_config_gate_action(**base)
    assert action == "refuse" and "Catalog #406" in message


def test_live_static_self_protect_gate_is_clean() -> None:
    assert check_launch_and_governor_require_dsl_compile_hash(repo_root=REPO) == []


def _copy_static_surfaces(tmp_path: Path) -> None:
    for rel in (
        "tools/launch_witness_run.py",
        "tools/spawn_durable_daemon.py",
        "tools/operator_authorize.py",
        "src/tac/admission_guard.py",
    ):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst)


def test_static_gate_detects_removed_governor_call(tmp_path: Path) -> None:
    _copy_static_surfaces(tmp_path)
    path = tmp_path / "tools/spawn_durable_daemon.py"
    path.write_text(path.read_text().replace(
        "refusal = _witness_dsl_compile_hash_gate(a, cmd)",
        "refusal = None",
        1,
    ))
    violations = check_launch_and_governor_require_dsl_compile_hash(repo_root=tmp_path)
    assert any("_do_start omits DSL governor gate" in row for row in violations)


def test_static_gate_respects_substantive_same_line_waiver(tmp_path: Path) -> None:
    _copy_static_surfaces(tmp_path)
    path = tmp_path / "tools/launch_witness_run.py"
    path.write_text(
        path.read_text()
        + "\ndef _reviewed_nonlaunch_probe():\n"
        + "    return subprocess.run('train_witness_probe')  # DSL_COMPILE_HASH_BYPASS_OK: reviewed non-launch static fixture\n"
    )
    assert check_launch_and_governor_require_dsl_compile_hash(repo_root=tmp_path) == []


def test_static_gate_rejects_placeholder_waiver(tmp_path: Path) -> None:
    _copy_static_surfaces(tmp_path)
    path = tmp_path / "tools/launch_witness_run.py"
    path.write_text(
        path.read_text()
        + "\ndef _bad_probe():\n"
        + "    return subprocess.run('train_witness_probe')  # DSL_COMPILE_HASH_BYPASS_OK:todo\n"
    )
    violations = check_launch_and_governor_require_dsl_compile_hash(repo_root=tmp_path)
    assert any("raw witness subprocess" in row for row in violations)


def test_native_witness_preflight_refuses_missing_binding() -> None:
    authorize = _load_tool("operator_authorize")
    recipe = SimpleNamespace(
        name="witness_dispatch",
        lane_id="witness_lane",
        remote_driver="experiments/train_witness.py",
        raw={},
    )
    with pytest.raises(SystemExit) as exc:
        authorize._native_dispatch_dsl_compile_hash_preflight(recipe)
    assert exc.value.code == 8


def test_actual_v9_identical_compile_is_deterministic(v9_document: dict) -> None:
    from tac.witness_dsl.spec_v9_cgauge import compile_v9_cgauge_ideal_launch_config

    cfg_b = compile_v9_cgauge_ideal_launch_config(
        "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
        num_pairs=600,
        mod_dim=19,
        program_name="v9_cgauge_ideal_mod19",
    )
    doc_b = build_dsl_compile_provenance_document(
        program_name=cfg_b.typed.name,
        typed_config=cfg_b.typed,
        compiler_manifest=cfg_b.constants_manifest,
        repo_root=REPO,
    )
    assert v9_document["dsl_compile_hash"] == doc_b["dsl_compile_hash"]


def test_internal_smoke_delta_is_typed_lever_and_writes_valid_binding(tmp_path: Path) -> None:
    from tac.witness_dsl.spec_v9_cgauge import compile_v9_cgauge_ideal_launch_config

    launcher = _load_tool("launch_witness_run")
    cfg = compile_v9_cgauge_ideal_launch_config(
        "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
        num_pairs=600,
        mod_dim=19,
        program_name="v9_cgauge_ideal_mod19",
    )
    rebound = launcher.with_internal_dsl_lever(
        cfg,
        name="catalog406_test_resume",
        overrides={"--ckpt-every": 1, "--resume-from": "prior/run"},
    )
    launch_sh, provenance, manifest, document = launcher.write_dsl_bound_launch(
        rebound, tmp_path
    )
    assert launch_sh.is_file() and provenance.is_file() and manifest.is_file()
    assert verify_dsl_provenance_artifacts(
        launch_sh, expected_hash=document["dsl_compile_hash"]
    )[0]
    bindings = {
        row["flag"]: row for row in document["bijection_manifest"]["bindings"]
    }
    assert bindings["--ckpt-every"]["lever_owners"] == ["catalog406_test_resume"]
    assert bindings["--resume-from"]["lever_owners"] == ["catalog406_test_resume"]


def test_actual_v9_run_identity_is_excluded_but_semantic_seed_is_not(
    v9_document: dict,
) -> None:
    from tac.witness_dsl.spec_v9_cgauge import compile_v9_cgauge_ideal_launch_config

    cfg = compile_v9_cgauge_ideal_launch_config(
        "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
        num_pairs=600,
        mod_dim=19,
        program_name="v9_cgauge_ideal_mod19",
    )
    run_identity_changed = cfg.typed.model_copy(
        update={"out_dir": "another/run/id", "purpose": "non-authoritative context"}
    )
    identity_doc = build_dsl_compile_provenance_document(
        program_name=cfg.typed.name,
        typed_config=run_identity_changed,
        compiler_manifest=cfg.constants_manifest,
        repo_root=REPO,
    )
    semantic_changed = cfg.typed.model_copy(update={"seed": cfg.typed.seed + 1})
    semantic_doc = build_dsl_compile_provenance_document(
        program_name=cfg.typed.name,
        typed_config=semantic_changed,
        compiler_manifest=cfg.constants_manifest,
        repo_root=REPO,
    )
    assert identity_doc["dsl_compile_hash"] == v9_document["dsl_compile_hash"]
    assert semantic_doc["dsl_compile_hash"] != v9_document["dsl_compile_hash"]
