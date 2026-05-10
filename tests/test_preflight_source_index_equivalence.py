from __future__ import annotations

from pathlib import Path

from tac import preflight
from tac.source_index import source_index_context


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _with_source_index(root: Path, fn):
    with source_index_context(root):
        return fn(repo_root=root, strict=False, verbose=False)


def test_comment_only_contract_source_index_matches_no_index(tmp_path: Path) -> None:
    _write(
        tmp_path / "scripts" / "remote_lane_demo.sh",
        """#!/usr/bin/env bash
# THE DEPLOY SCRIPT CALLS the real trainer later.
python - <<'PY'
print("stub")
PY
""",
    )
    _write(
        tmp_path / "src" / "tac" / "guarded_contract.py",
        '''def guarded() -> None:
    """The deploy script calls the real implementation."""
    raise RuntimeError("wrapper did not replace guarded")
''',
    )

    no_index = preflight.check_no_comment_only_contracts(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    with_index = _with_source_index(tmp_path, preflight.check_no_comment_only_contracts)

    assert with_index == no_index
    assert len(with_index) == 1
    assert "scripts/remote_lane_demo.sh:2" in with_index[0]


def test_bare_round_roundtrip_source_index_matches_no_index(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "tac" / "roundtrip_bad.py",
        """import torch.nn.functional as F

def eval_roundtrip(x):
    y = F.interpolate(x, scale_factor=2)
    return y.round()
""",
    )
    _write(
        tmp_path / "experiments" / "roundtrip_good.py",
        """import torch.nn.functional as F

def eval_roundtrip_ok(x):
    y = F.interpolate(x, scale_factor=2)
    return y.round() + (y - y.round()).detach()
""",
    )

    no_index = preflight.check_no_bare_round_in_eval_roundtrip(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    with_index = _with_source_index(tmp_path, preflight.check_no_bare_round_in_eval_roundtrip)

    assert with_index == no_index
    assert len(with_index) == 1
    assert "src/tac/roundtrip_bad.py:5" in with_index[0]


def test_profile_resolver_source_index_matches_no_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE", "1")
    monkeypatch.setattr(
        preflight,
        "_extract_profile_keys",
        lambda: {"alpha_key", "beta_key"},
    )
    _write(tmp_path / "src" / "tac" / "profiles.py", "PROFILES = {}\n")
    _write(
        tmp_path / "experiments" / "train_demo.py",
        """def train(alpha_key: int = 1) -> int:
    return alpha_key
""",
    )
    _write(
        tmp_path / "src" / "tac" / "irrelevant.py",
        "VALUE = 'no profile keys here'\n",
    )

    no_index = preflight.check_profile_keys_have_resolvers(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    with_index = _with_source_index(tmp_path, preflight.check_profile_keys_have_resolvers)

    assert with_index == no_index
    assert len(with_index) == 1
    assert "beta_key" in with_index[0]
