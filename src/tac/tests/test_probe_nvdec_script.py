from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "probe_nvdec.sh"


def test_probe_nvdec_ensure_dali_uses_hash_pinned_bootstrap_before_pip() -> None:
    text = SCRIPT.read_text()
    assert "bootstrap_dali_hash_pinned.py" in text
    assert "DALI_BOOTSTRAP_JSON" in text
    assert "--json-out" in text
    assert "--force" in text
    assert 'elif "$PYBIN" -m pip --version' in text
    assert 'uv pip install --python "$PYBIN"' in text
