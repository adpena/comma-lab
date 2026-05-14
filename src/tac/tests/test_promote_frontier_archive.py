"""Tests for tools/promote_frontier_archive.py.

Covers F2 fix per A1 PR Council Round 1: dual-axis paired evidence promotion
discipline + each refusal class.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


# Load tools/promote_frontier_archive.py via importlib so the test does not
# require ``tools`` to be a package.
def _load_promote_module():
    spec = importlib.util.spec_from_file_location(
        "promote_frontier_archive_under_test",
        REPO_ROOT / "tools" / "promote_frontier_archive.py",
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["promote_frontier_archive_under_test"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


promote_mod = _load_promote_module()
FrontierPromotionRefused = promote_mod.FrontierPromotionRefused
parse_axis_evidence = promote_mod.parse_axis_evidence
promote_archive = promote_mod.promote_archive
build_integrity_manifest = promote_mod.build_integrity_manifest
render_serializer_command = promote_mod.render_serializer_command
_classify_evidence_grade = promote_mod._classify_evidence_grade


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_archive(path: Path, payload: bytes = b"hello-world-archive") -> str:
    """Create a tiny zip archive at ``path``; return its sha256 hex."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("0.bin", payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inflate_sh(path: Path) -> None:
    path.write_text("#!/bin/sh\nset -euo pipefail\necho ok\n", encoding="utf-8")
    path.chmod(0o755)


def _cpu_eval_payload(*, sha256: str, score: float = 0.193) -> dict[str, Any]:
    return {
        "archive_sha256": sha256,
        "archive_size_bytes": 178000,
        "canonical_score_recomputed": score,
        "n_samples": 600,
        "device": "cpu",
        "hardware": "github-actions-ubuntu-latest-x86_64",
        "lane_tag": "[contest-CPU GHA Linux x86_64]",
        "evidence_grade": "contest-CPU-1to1",
    }


def _cuda_eval_payload(*, sha256: str, score: float = 0.226) -> dict[str, Any]:
    return {
        "archive_sha256": sha256,
        "archive_size_bytes": 178000,
        "canonical_score_recomputed": score,
        "n_samples": 600,
        "device": "cuda",
        "hardware": "Tesla T4 (Vast.ai)",
        "lane_tag": "[contest-CUDA]",
        "evidence_grade": "contest-CUDA",
    }


@pytest.fixture
def stage(tmp_path: Path):
    """Stage a real archive + inflate.sh + paired CPU/CUDA evals under ``tmp_path``."""
    src = tmp_path / "src_results"
    src.mkdir()
    archive = src / "archive.zip"
    sha = _make_archive(archive)
    inflate_sh = src / "inflate.sh"
    _write_inflate_sh(inflate_sh)
    cpu_path = src / "contest_auth_eval.cpu.json"
    cpu_path.write_text(json.dumps(_cpu_eval_payload(sha256=sha)), encoding="utf-8")
    cuda_path = src / "contest_auth_eval.cuda.json"
    cuda_path.write_text(json.dumps(_cuda_eval_payload(sha256=sha)), encoding="utf-8")

    target = tmp_path / "submissions_frontier" / "test_label"
    return {
        "archive_path": archive,
        "archive_sha256": sha,
        "inflate_sh": inflate_sh,
        "cpu_eval_path": cpu_path,
        "cuda_eval_path": cuda_path,
        "target_dir": target,
        "tmp_path": tmp_path,
    }


# ── 1. parse_axis_evidence happy path ─────────────────────────────────────


def test_parse_axis_evidence_cpu_payload(stage):
    ev = parse_axis_evidence(stage["cpu_eval_path"], axis="cpu")
    assert ev.axis == "cpu"
    assert ev.score_value == pytest.approx(0.193)
    assert ev.evidence_tag == "[contest-CPU GHA Linux x86_64]"
    assert ev.hardware_substrate == "linux_x86_64_gha_cpu"
    assert ev.archive_sha256 == stage["archive_sha256"]
    assert ev.n_samples == 600


def test_parse_axis_evidence_cuda_payload(stage):
    ev = parse_axis_evidence(stage["cuda_eval_path"], axis="cuda")
    assert ev.axis == "cuda"
    assert ev.evidence_tag == "[contest-CUDA]"
    assert ev.hardware_substrate == "linux_x86_64_t4"


def test_parse_axis_evidence_4090_hardware(stage, tmp_path):
    payload = _cuda_eval_payload(sha256=stage["archive_sha256"])
    payload["hardware"] = "RTX 4090"
    p = tmp_path / "cuda_4090.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    ev = parse_axis_evidence(p, axis="cuda")
    assert ev.hardware_substrate == "linux_x86_64_4090"


def test_parse_axis_evidence_modal_cpu(stage, tmp_path):
    payload = _cpu_eval_payload(sha256=stage["archive_sha256"])
    payload["hardware"] = "modal-cpu-shared-x86_64"
    p = tmp_path / "modal_cpu.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    ev = parse_axis_evidence(p, axis="cpu")
    assert ev.hardware_substrate == "linux_x86_64_modal_cpu"


# ── 2. parse_axis_evidence refusal classes ────────────────────────────────


def test_parse_refuses_when_eval_missing(tmp_path):
    with pytest.raises(FrontierPromotionRefused) as excinfo:
        parse_axis_evidence(tmp_path / "nonexistent.json", axis="cpu")
    assert excinfo.value.refusal_class == "eval_json_missing"


def test_parse_refuses_when_unparseable(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not-json", encoding="utf-8")
    with pytest.raises(FrontierPromotionRefused) as excinfo:
        parse_axis_evidence(p, axis="cpu")
    assert excinfo.value.refusal_class == "eval_json_unparseable"


def test_parse_refuses_when_no_evidence_tag(tmp_path):
    p = tmp_path / "no_tag.json"
    p.write_text(
        json.dumps({"archive_sha256": "x" * 64, "canonical_score_recomputed": 0.2}),
        encoding="utf-8",
    )
    with pytest.raises(FrontierPromotionRefused) as excinfo:
        parse_axis_evidence(p, axis="cpu")
    assert excinfo.value.refusal_class == "evidence_tag_missing"


def test_parse_refuses_when_no_score(tmp_path):
    p = tmp_path / "no_score.json"
    p.write_text(
        json.dumps({"archive_sha256": "x" * 64, "lane_tag": "[contest-CUDA]"}),
        encoding="utf-8",
    )
    with pytest.raises(FrontierPromotionRefused) as excinfo:
        parse_axis_evidence(p, axis="cuda")
    assert excinfo.value.refusal_class == "score_value_missing"


def test_parse_refuses_when_no_archive_sha(tmp_path):
    p = tmp_path / "no_sha.json"
    p.write_text(
        json.dumps(
            {"canonical_score_recomputed": 0.2, "lane_tag": "[contest-CUDA]"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(FrontierPromotionRefused) as excinfo:
        parse_axis_evidence(p, axis="cuda")
    assert excinfo.value.refusal_class == "archive_sha256_field_missing"


# ── 3. _classify_evidence_grade ───────────────────────────────────────────


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("[contest-CUDA]", None),
        ("[contest-CPU GHA Linux x86_64]", None),
        ("[contest-CPU GHA]", None),
        ("[contest-CPU]", None),
        ("[macOS-CPU advisory only]", "macos_advisory_grade"),
        ("[macOS-CPU calibrated]", "macos_advisory_grade"),
        ("[MPS-PROXY]", "mps_evidence"),
        ("[MPS-research-signal]", "mps_evidence"),
        ("[advisory only]", "advisory_grade"),
        ("[byte-anchor]", "advisory_grade"),
        ("[some weird unrecognized tag]", "advisory_grade"),
        ("", "evidence_tag_missing"),
    ],
)
def test_classify_evidence_grade(tag, expected):
    assert _classify_evidence_grade(tag) == expected


# ── 4. promote_archive happy path ─────────────────────────────────────────


def test_promote_archive_happy_path(stage):
    manifest = promote_archive(
        archive_path=stage["archive_path"],
        inflate_sh_path=stage["inflate_sh"],
        contest_cpu_eval_path=stage["cpu_eval_path"],
        contest_cuda_eval_path=stage["cuda_eval_path"],
        label="test_label",
        target_dir=stage["target_dir"],
        operator="testbot",
        notes="happy-path test",
        repo_root=stage["tmp_path"],
    )
    assert manifest["label"] == "test_label"
    assert manifest["archive"]["sha256"] == stage["archive_sha256"]
    assert manifest["axes"]["cpu"]["score_value"] == pytest.approx(0.193)
    assert manifest["axes"]["cuda"]["score_value"] == pytest.approx(0.226)
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is True
    target = stage["target_dir"]
    assert (target / "archive.zip").is_file()
    assert (target / "inflate.sh").is_file()
    assert (target / "integrity_manifest.json").is_file()
    assert (target / "promotion_provenance.json").is_file()
    # Archive bytes must round-trip exactly.
    src_bytes = stage["archive_path"].read_bytes()
    tgt_bytes = (target / "archive.zip").read_bytes()
    assert src_bytes == tgt_bytes


def test_promote_archive_writes_inflate_py_when_present(stage):
    py = stage["archive_path"].parent / "inflate.py"
    py.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    promote_archive(
        archive_path=stage["archive_path"],
        inflate_sh_path=stage["inflate_sh"],
        inflate_py_path=py,
        contest_cpu_eval_path=stage["cpu_eval_path"],
        contest_cuda_eval_path=stage["cuda_eval_path"],
        label="with_py",
        target_dir=stage["target_dir"],
        repo_root=stage["tmp_path"],
    )
    assert (stage["target_dir"] / "inflate.py").is_file()


# ── 5. promote_archive refusal classes ────────────────────────────────────


def test_refuses_on_archive_missing(stage):
    with pytest.raises(FrontierPromotionRefused) as excinfo:
        promote_archive(
            archive_path=stage["tmp_path"] / "no-such.zip",
            inflate_sh_path=stage["inflate_sh"],
            contest_cpu_eval_path=stage["cpu_eval_path"],
            contest_cuda_eval_path=stage["cuda_eval_path"],
            label="x",
            target_dir=stage["target_dir"],
            repo_root=stage["tmp_path"],
        )
    assert excinfo.value.refusal_class == "archive_path_missing"


def test_refuses_on_inflate_sh_missing(stage):
    with pytest.raises(FrontierPromotionRefused) as excinfo:
        promote_archive(
            archive_path=stage["archive_path"],
            inflate_sh_path=stage["tmp_path"] / "no-inflate.sh",
            contest_cpu_eval_path=stage["cpu_eval_path"],
            contest_cuda_eval_path=stage["cuda_eval_path"],
            label="x",
            target_dir=stage["target_dir"],
            repo_root=stage["tmp_path"],
        )
    assert excinfo.value.refusal_class == "inflate_sh_missing"


def test_refuses_on_macos_advisory(stage):
    payload = _cpu_eval_payload(sha256=stage["archive_sha256"])
    payload["lane_tag"] = "[macOS-CPU advisory only]"
    payload["hardware"] = "macos_arm64"
    stage["cpu_eval_path"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FrontierPromotionRefused) as excinfo:
        promote_archive(
            archive_path=stage["archive_path"],
            inflate_sh_path=stage["inflate_sh"],
            contest_cpu_eval_path=stage["cpu_eval_path"],
            contest_cuda_eval_path=stage["cuda_eval_path"],
            label="x",
            target_dir=stage["target_dir"],
            repo_root=stage["tmp_path"],
        )
    assert excinfo.value.refusal_class == "macos_advisory_grade"


def test_refuses_on_mps_evidence(stage):
    payload = _cuda_eval_payload(sha256=stage["archive_sha256"])
    payload["lane_tag"] = "[MPS-PROXY]"
    stage["cuda_eval_path"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FrontierPromotionRefused) as excinfo:
        promote_archive(
            archive_path=stage["archive_path"],
            inflate_sh_path=stage["inflate_sh"],
            contest_cpu_eval_path=stage["cpu_eval_path"],
            contest_cuda_eval_path=stage["cuda_eval_path"],
            label="x",
            target_dir=stage["target_dir"],
            repo_root=stage["tmp_path"],
        )
    assert excinfo.value.refusal_class == "mps_evidence"


def test_refuses_on_archive_sha_mismatch(stage):
    """CPU and CUDA evals reference different archives → apples-to-apples violation."""
    payload = _cuda_eval_payload(sha256="0" * 64)  # different sha
    stage["cuda_eval_path"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FrontierPromotionRefused) as excinfo:
        promote_archive(
            archive_path=stage["archive_path"],
            inflate_sh_path=stage["inflate_sh"],
            contest_cpu_eval_path=stage["cpu_eval_path"],
            contest_cuda_eval_path=stage["cuda_eval_path"],
            label="x",
            target_dir=stage["target_dir"],
            repo_root=stage["tmp_path"],
        )
    assert excinfo.value.refusal_class == "archive_sha256_mismatch"


def test_refuses_on_hardware_not_contest_compliant(stage):
    """CPU eval with a non-Linux-x86_64 substrate (e.g. an exotic ARM CPU)."""
    payload = _cpu_eval_payload(sha256=stage["archive_sha256"])
    payload["hardware"] = "raspberry_pi_4"  # unrecognized → empty after heuristic
    stage["cpu_eval_path"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FrontierPromotionRefused) as excinfo:
        promote_archive(
            archive_path=stage["archive_path"],
            inflate_sh_path=stage["inflate_sh"],
            contest_cpu_eval_path=stage["cpu_eval_path"],
            contest_cuda_eval_path=stage["cuda_eval_path"],
            label="x",
            target_dir=stage["target_dir"],
            repo_root=stage["tmp_path"],
        )
    assert excinfo.value.refusal_class == "hardware_not_contest_compliant"


def test_refuses_on_target_dir_outside_repo(stage, tmp_path):
    """Refuse if target dir resolves outside the declared repo root."""
    other_root = tmp_path / "other_repo"
    other_root.mkdir()
    with pytest.raises(FrontierPromotionRefused) as excinfo:
        promote_archive(
            archive_path=stage["archive_path"],
            inflate_sh_path=stage["inflate_sh"],
            contest_cpu_eval_path=stage["cpu_eval_path"],
            contest_cuda_eval_path=stage["cuda_eval_path"],
            label="x",
            target_dir=tmp_path / "target_outside",
            repo_root=other_root,
        )
    assert excinfo.value.refusal_class == "target_dir_not_under_repo"


def test_refuses_on_target_dir_collision_without_overwrite(stage):
    """Refuse if target dir non-empty and --overwrite not set."""
    stage["target_dir"].mkdir(parents=True)
    (stage["target_dir"] / "preexisting.txt").write_text("hi", encoding="utf-8")
    with pytest.raises(FrontierPromotionRefused) as excinfo:
        promote_archive(
            archive_path=stage["archive_path"],
            inflate_sh_path=stage["inflate_sh"],
            contest_cpu_eval_path=stage["cpu_eval_path"],
            contest_cuda_eval_path=stage["cuda_eval_path"],
            label="x",
            target_dir=stage["target_dir"],
            repo_root=stage["tmp_path"],
        )
    assert excinfo.value.refusal_class == "target_path_collision"


def test_overwrite_allows_collision(stage):
    """--overwrite lets us replace an existing target dir."""
    stage["target_dir"].mkdir(parents=True)
    (stage["target_dir"] / "preexisting.txt").write_text("hi", encoding="utf-8")
    manifest = promote_archive(
        archive_path=stage["archive_path"],
        inflate_sh_path=stage["inflate_sh"],
        contest_cpu_eval_path=stage["cpu_eval_path"],
        contest_cuda_eval_path=stage["cuda_eval_path"],
        label="x",
        target_dir=stage["target_dir"],
        overwrite=True,
        repo_root=stage["tmp_path"],
    )
    assert manifest["label"] == "x"


def test_axis_score_diff_above_noise_blocks(stage):
    """If CPU and CUDA scores differ by > noise_floor, refuse with axis_score_diff."""
    payload = _cpu_eval_payload(sha256=stage["archive_sha256"])
    payload["canonical_score_recomputed"] = 0.05  # CUDA stays at 0.226 → diff 0.176
    stage["cpu_eval_path"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FrontierPromotionRefused) as excinfo:
        promote_archive(
            archive_path=stage["archive_path"],
            inflate_sh_path=stage["inflate_sh"],
            contest_cpu_eval_path=stage["cpu_eval_path"],
            contest_cuda_eval_path=stage["cuda_eval_path"],
            label="x",
            target_dir=stage["target_dir"],
            cpu_cuda_noise_floor=0.05,
            repo_root=stage["tmp_path"],
        )
    assert excinfo.value.refusal_class == "axis_score_diff_above_noise"


# ── 6. provenance log appends ─────────────────────────────────────────────


def test_promotion_provenance_appends_on_repromotion(stage):
    promote_archive(
        archive_path=stage["archive_path"],
        inflate_sh_path=stage["inflate_sh"],
        contest_cpu_eval_path=stage["cpu_eval_path"],
        contest_cuda_eval_path=stage["cuda_eval_path"],
        label="x",
        target_dir=stage["target_dir"],
        repo_root=stage["tmp_path"],
    )
    promote_archive(
        archive_path=stage["archive_path"],
        inflate_sh_path=stage["inflate_sh"],
        contest_cpu_eval_path=stage["cpu_eval_path"],
        contest_cuda_eval_path=stage["cuda_eval_path"],
        label="x",
        target_dir=stage["target_dir"],
        overwrite=True,
        repo_root=stage["tmp_path"],
    )
    provenance = json.loads(
        (stage["target_dir"] / "promotion_provenance.json").read_text(encoding="utf-8")
    )
    assert len(provenance["attempts"]) == 2


# ── 7. render_serializer_command ──────────────────────────────────────────


def test_render_serializer_command_includes_canonical_pieces(stage):
    promote_archive(
        archive_path=stage["archive_path"],
        inflate_sh_path=stage["inflate_sh"],
        contest_cpu_eval_path=stage["cpu_eval_path"],
        contest_cuda_eval_path=stage["cuda_eval_path"],
        label="my_label",
        target_dir=stage["target_dir"],
        repo_root=stage["tmp_path"],
    )
    cmd = render_serializer_command(
        stage["target_dir"], "my_label", repo_root=stage["tmp_path"]
    )
    assert "tools/subagent_commit_serializer.py" in cmd
    assert "--expected-content-sha256" in cmd
    assert "--reason" in cmd
    assert "frontier promotion: my_label" in cmd
    assert "archive.zip" in cmd
    assert "integrity_manifest.json" in cmd


# ── 8. integrity manifest schema invariants ───────────────────────────────


def test_integrity_manifest_carries_compliance_tags(stage):
    manifest = promote_archive(
        archive_path=stage["archive_path"],
        inflate_sh_path=stage["inflate_sh"],
        contest_cpu_eval_path=stage["cpu_eval_path"],
        contest_cuda_eval_path=stage["cuda_eval_path"],
        label="x",
        target_dir=stage["target_dir"],
        repo_root=stage["tmp_path"],
    )
    tags = manifest["claude_md_compliance_tags"]
    assert "submission_auth_eval_dual_axis_1to1_contest_compliant" in tags
    assert "apples_to_apples_paired_archive_sha256" in tags
    assert "no_mps_authoritative" in tags
    assert "no_macos_cpu_authoritative" in tags
    assert "no_score_claim_advanced_by_this_artifact" in tags


def test_integrity_manifest_score_claim_is_false(stage):
    manifest = promote_archive(
        archive_path=stage["archive_path"],
        inflate_sh_path=stage["inflate_sh"],
        contest_cpu_eval_path=stage["cpu_eval_path"],
        contest_cuda_eval_path=stage["cuda_eval_path"],
        label="x",
        target_dir=stage["target_dir"],
        repo_root=stage["tmp_path"],
    )
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
