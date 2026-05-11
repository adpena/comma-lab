"""Tests for the PR103-on-PR106 final runtime packet."""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) in sys.path:
        sys.path.remove(str(_path))
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_submissions = sys.modules.get("submissions")
if _submissions is not None:
    expected = (REPO_ROOT / "submissions").resolve()
    package_paths = {
        Path(path).resolve() for path in getattr(_submissions, "__path__", ())
    }
    module_file = getattr(_submissions, "__file__", None)
    file_ok = module_file is not None and expected in Path(module_file).resolve().parents
    if expected not in package_paths and not file_ok:
        for _name in list(sys.modules):
            if _name == "submissions" or _name.startswith("submissions."):
                del sys.modules[_name]

from submissions.pr103_pr106_final_runtime import inflate as runtime
from tac.pr103_pr106_runtime_closure import (
    Pr103Pr106RuntimeClosure,
    parse_pr103_repacked_pr106_payload,
)

CANDIDATE_ARCHIVE = (
    REPO_ROOT / "experiments/results/pr103_repack_pr106_standalone_20260507/archive.zip"
)
RUNTIME_CLOSURE_JSON = (
    REPO_ROOT / "experiments/results/pr103_repack_pr106_standalone_20260507/runtime_closure.json"
)
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _candidate_payload() -> bytes:
    with zipfile.ZipFile(CANDIDATE_ARCHIVE) as zf:
        return zf.read("0.bin")


@pytest.mark.skipif(not RUNTIME_CLOSURE_JSON.is_file(), reason="runtime closure artifact missing")
def test_final_runtime_embeds_existing_closure_artifact() -> None:
    record = json.loads(RUNTIME_CLOSURE_JSON.read_text())
    assert record["runtime_closure"] == runtime.RUNTIME_CLOSURE


@pytest.mark.skipif(
    not CANDIDATE_ARCHIVE.is_file() or not RUNTIME_CLOSURE_JSON.is_file(),
    reason="candidate archive and runtime closure artifact required",
)
def test_final_runtime_decodes_candidate_like_tac_closure() -> None:
    payload = _candidate_payload()
    runtime_sd, runtime_latents, runtime_meta = runtime.parse_pr103_pr106_archive(payload)
    closure_record = json.loads(RUNTIME_CLOSURE_JSON.read_text())
    tac_decoded = parse_pr103_repacked_pr106_payload(
        payload,
        Pr103Pr106RuntimeClosure.from_dict(closure_record["runtime_closure"]),
    )

    assert runtime_meta == tac_decoded.meta
    assert tuple(runtime_latents.shape) == (600, 28)
    assert (runtime_latents - tac_decoded.latents).abs().max().item() == 0.0
    assert set(runtime_sd) == set(tac_decoded.state_dict)
    for name, tensor in runtime_sd.items():
        assert tensor.shape == tac_decoded.state_dict[name].shape
        assert (tensor - tac_decoded.state_dict[name]).abs().max().item() == 0.0


@pytest.mark.skipif(not CANDIDATE_ARCHIVE.is_file(), reason="candidate archive required")
def test_final_runtime_fails_closed_on_payload_mismatch() -> None:
    payload = bytearray(_candidate_payload())
    payload[4] ^= 0x01
    with pytest.raises(runtime.RuntimeClosureError, match="payload SHA-256 mismatch"):
        runtime.parse_pr103_pr106_archive(bytes(payload))


def test_final_runtime_dependency_check_covers_brotli_and_constriction() -> None:
    versions = runtime.runtime_dependency_versions()
    assert versions["brotli"]
    assert versions["constriction"]


def test_final_runtime_selects_cpu_when_cuda_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: False)

    assert runtime.select_inflate_device().type == "cpu"


def test_final_runtime_selects_cuda_when_available(monkeypatch) -> None:
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)

    assert runtime.select_inflate_device().type == "cuda"


def test_final_runtime_inflate_device_override_is_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("PACT_INFLATE_DEVICE", "cpu")

    assert runtime.select_inflate_device().type == "cpu"

    monkeypatch.setenv("PACT_INFLATE_DEVICE", "cuda")
    assert runtime.select_inflate_device().type == "cuda"

    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: False)
    with pytest.raises(runtime.RuntimeClosureError, match="CUDA is unavailable"):
        runtime.select_inflate_device()

    monkeypatch.setenv("PACT_INFLATE_DEVICE", "bogus")
    with pytest.raises(runtime.RuntimeClosureError, match="invalid PACT_INFLATE_DEVICE"):
        runtime.select_inflate_device()


def test_pyproject_keeps_brotli_and_constriction_as_hard_runtime_deps() -> None:
    text = PYPROJECT.read_text()
    assert re.search(r'"brotli>=1\.0"', text)
    assert re.search(r'"constriction>=0\.4,<0\.5"', text)


def test_final_runtime_inflate_sh_is_upload_root_safe() -> None:
    text = (REPO_ROOT / "submissions/pr103_pr106_final_runtime/inflate.sh").read_text()
    assert 'LOCAL_INFLATE_PY="$HERE/inflate.py"' in text
    assert 'PY_INFLATE=( "$PYBIN" "$LOCAL_INFLATE_PY" )' in text
    assert '"${PY_INFLATE[@]}" --dependency-check' in text
    assert '"${PY_INFLATE[@]}" "$SRC" "$DST"' in text


def test_final_runtime_has_no_tac_or_scorer_import_surface() -> None:
    text = (REPO_ROOT / "submissions/pr103_pr106_final_runtime/inflate.py").read_text().lower()
    assert "from tac." not in text
    assert "import tac." not in text
    assert "upstream/evaluate" not in text
    assert "tac.scorers" not in text
