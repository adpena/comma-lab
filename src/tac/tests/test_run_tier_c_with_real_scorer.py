# SPDX-License-Identifier: MIT
"""Tests for tools/run_tier_c_with_real_scorer.py.

The runner's production path loads real SegNet+PoseNet weights, so tests mock
that boundary and verify the orchestration contract: fail closed by default,
load scorer once in execute mode, and never emit score-claim authority.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "run_tier_c_with_real_scorer.py"


def _load_runner():
    name = "_run_tier_c_with_real_scorer_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_zip(path: Path, member: str = "x", payload: bytes = b"payload") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, payload)
    return path


def test_default_candidates_cover_real_scorer_controls() -> None:
    mod = _load_runner()
    names = {spec.name for spec in mod.DEFAULT_CANDIDATES}
    grammars = {spec.grammar for spec in mod.DEFAULT_CANDIDATES}
    assert names == {
        "a1",
        "pr106_r2",
        "ibps1_c6_5ep",
        "ibps1_c6_100ep_a10g_advisory",
        "dp1_smoke",
    }
    assert grammars == {"a1", "pr106_latent_sidecar", "ibps1", "dp1"}


def test_parse_archive_spec_requires_grammar(tmp_path: Path) -> None:
    mod = _load_runner()
    archive = _write_zip(tmp_path / "archive.zip")
    spec = mod._parse_archive_spec(f"custom={archive},grammar=dp1,role=test", REPO)
    assert spec.name == "custom"
    assert spec.archive_path == archive
    assert spec.grammar == "dp1"
    assert spec.role == "test"
    with pytest.raises(ValueError, match="missing grammar"):
        mod._parse_archive_spec(f"custom={archive}", REPO)


def test_plan_mode_writes_non_claim_manifest(tmp_path: Path) -> None:
    mod = _load_runner()
    archive = _write_zip(tmp_path / "archive.zip")
    out_dir = tmp_path / "out"
    rc = mod.main(
        [
            "--output-dir",
            str(out_dir),
            "--upstream-dir",
            str(tmp_path / "missing_upstream"),
            "--archive",
            f"a1={archive},grammar=a1",
        ]
    )
    assert rc == 0
    payload = json.loads((out_dir / "tier_c_real_scorer_manifest.json").read_text())
    assert payload["mode"] == "plan_only"
    assert payload["score_claim"] is False
    assert payload["score_claim_valid"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["candidate_specs"][0]["archive_exists"] is True
    assert "missing_real_scorer_assets" in payload["blockers"]


def test_execute_fails_closed_when_scorer_assets_missing(tmp_path: Path) -> None:
    mod = _load_runner()
    archive = _write_zip(tmp_path / "archive.zip")
    out_dir = tmp_path / "out"
    rc = mod.main(
        [
            "--execute",
            "--output-dir",
            str(out_dir),
            "--upstream-dir",
            str(tmp_path / "missing_upstream"),
            "--archive",
            f"a1={archive},grammar=a1",
        ]
    )
    assert rc == 2
    payload = json.loads((out_dir / "tier_c_real_scorer_manifest.json").read_text())
    assert payload["mode"] == "failed_closed"
    assert payload["score_claim"] is False
    assert payload["error"]["class"] == "fail_closed_prerequisite_error"
    assert "missing_real_scorer_assets" in payload["blockers"]


@dataclasses.dataclass
class _FakeTierCResult:
    target: str
    noise_sigma_relative: float
    delta_seg: float
    delta_pose: float
    delta_score_components: float
    elapsed_seconds: float


@dataclasses.dataclass
class _FakeArchiveAblationResult:
    archive_name: str
    archive_path: str
    archive_sha256: str
    archive_size_bytes: int
    grammar: str
    device: str
    pair_samples: int
    baseline_seg: float
    baseline_pose: float
    baseline_score_components: float
    pair_indices: list[int]
    decision_grade: bool
    included_sections: list[str]
    excluded_sections: list[str]
    timestamp_utc: str
    notes: list[str]
    tier_c: list[_FakeTierCResult] = dataclasses.field(default_factory=list)
    mdl_tier_c_density_estimate: float = 0.0
    mdl_tier_c_substrate_class_verdict: str = ""
    mdl_tier_c_curve_knee_signal: float = 0.0
    mdl_tier_c_latent_sigma1_delta: float = 0.0


class _FakeTorch:
    def __init__(self) -> None:
        self.seeds: list[int] = []

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def device(self, name: str) -> str:
        assert name == "cpu"
        return name


def _fake_mdl(*, decode_raises_for_dp1: bool = False) -> SimpleNamespace:
    torch = _FakeTorch()
    calls: dict[str, object] = {"run_tier_c": 0, "noise_sigmas": []}

    def aggregate(result):
        result.mdl_tier_c_density_estimate = 0.25
        result.mdl_tier_c_substrate_class_verdict = "across_class"
        result.mdl_tier_c_curve_knee_signal = 0.75
        result.mdl_tier_c_latent_sigma1_delta = 0.01
        return result

    def run_tier_c(*_args, **_kwargs):
        calls["run_tier_c"] = int(calls["run_tier_c"]) + 1
        calls["noise_sigmas"].append(_kwargs.get("noise_sigmas"))  # type: ignore[union-attr]
        return [
            _FakeTierCResult("state_dict", 0.01, 0.1, 0.2, 0.3, 0.01),
            _FakeTierCResult("latents", 1.0, 0.01, 0.02, 0.03, 0.01),
        ]

    def decode_to_frames(_inner, grammar, *_args, **_kwargs):
        if decode_raises_for_dp1 and grammar == "dp1":
            raise NotImplementedError("grammar dp1 not supported")
        return "frames"

    return SimpleNamespace(
        torch=torch,
        np=SimpleNamespace(random=SimpleNamespace(seed=lambda _seed: None)),
        calls=calls,
        normalize_grammar=lambda grammar: grammar,
        load_archive=lambda _path, _grammar: (b"inner", {"x": (0, 5)}),
        _load_ground_truth_pairs=lambda _video, _pairs: "gt_pairs",
        decode_to_frames=decode_to_frames,
        _compute_seg_pose_delta=lambda *_args, **_kwargs: (0.0002, 0.0003),
        _score_components=lambda pose, seg: 100.0 * seg + pose,
        run_tier_c=run_tier_c,
        aggregate_mdl_estimate=aggregate,
        ArchiveAblationResult=_FakeArchiveAblationResult,
    )


def test_execute_uses_one_scorer_load_for_multiple_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_runner()
    upstream = tmp_path / "upstream"
    for rel in (
        "modules.py",
        "models/posenet.safetensors",
        "models/segnet.safetensors",
        "videos/0.mkv",
    ):
        p = upstream / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"asset")
    a1 = _write_zip(tmp_path / "a1.zip")
    dp1 = _write_zip(tmp_path / "dp1.zip", member="0.bin")
    fake_mdl = _fake_mdl()
    scorer_loads = {"count": 0}

    monkeypatch.setattr(mod, "_load_mdl_module", lambda: fake_mdl)

    def fake_load_scorer(_mdl, _upstream):
        scorer_loads["count"] += 1
        return "real_scorer_stub"

    monkeypatch.setattr(mod, "_load_real_scorer", fake_load_scorer)

    out_dir = tmp_path / "out"
    rc = mod.main(
        [
            "--execute",
            "--output-dir",
            str(out_dir),
            "--upstream-dir",
            str(upstream),
            "--pair-samples",
            "2",
            "--archive",
            f"a1={a1},grammar=a1",
            "--archive",
            f"dp1={dp1},grammar=dp1",
        ]
    )
    assert rc == 0
    assert scorer_loads["count"] == 1
    assert fake_mdl.calls["run_tier_c"] == 2

    payload = json.loads((out_dir / "tier_c_real_scorer_manifest.json").read_text())
    assert payload["mode"] == "executed"
    assert payload["score_claim"] is False
    assert payload["score_claim_valid"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert len(payload["archives"]) == 2
    assert all(row["mdl_tier_c_substrate_class_verdict"] == "across_class" for row in payload["archives"])
    assert (out_dir / "a1_tier_c_real_scorer.json").is_file()
    assert (out_dir / "dp1_tier_c_real_scorer.json").is_file()


def test_execute_dp1_uses_zero_sigma_baseline_when_decoder_dispatcher_lacks_dp1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_runner()
    upstream = tmp_path / "upstream"
    for rel in (
        "modules.py",
        "models/posenet.safetensors",
        "models/segnet.safetensors",
        "videos/0.mkv",
    ):
        p = upstream / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"asset")
    dp1 = _write_zip(tmp_path / "dp1.zip", member="0.bin")
    fake_mdl = _fake_mdl(decode_raises_for_dp1=True)
    monkeypatch.setattr(mod, "_load_mdl_module", lambda: fake_mdl)
    monkeypatch.setattr(mod, "_load_real_scorer", lambda _mdl, _upstream: "scorer")

    out_dir = tmp_path / "out"
    rc = mod.main(
        [
            "--execute",
            "--output-dir",
            str(out_dir),
            "--upstream-dir",
            str(upstream),
            "--pair-samples",
            "1",
            "--archive",
            f"dp1={dp1},grammar=dp1",
        ]
    )
    assert rc == 0
    assert fake_mdl.calls["run_tier_c"] == 2
    assert fake_mdl.calls["noise_sigmas"][0] == [0.0]  # type: ignore[index]
    assert fake_mdl.calls["noise_sigmas"][1] is None  # type: ignore[index]
