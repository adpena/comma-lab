"""Integration controls for the live custody preflight wrapper."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

from tac.preflight import PreflightError, check_moved_payload_destinations


def test_live_discovery_strict_and_warning(tmp_path):
    data = tmp_path / 'payload'
    data.write_bytes(b'actual payload')
    cert = tmp_path / 'source.MOVED.json'
    cert.write_text(json.dumps({'moved_to': str(data), 'bytes': 14,
                               'sha256': hashlib.sha256(data.read_bytes()).hexdigest()}))
    assert check_moved_payload_destinations(roots=[tmp_path], strict=True) == []
    data.unlink()
    assert check_moved_payload_destinations(roots=[tmp_path])
    with pytest.raises(PreflightError, match='419'):
        check_moved_payload_destinations(roots=[tmp_path], strict=True)


def test_empty_scope_is_not_pass(tmp_path):
    assert 'NO_COVERAGE' in check_moved_payload_destinations(roots=[tmp_path])[0]
    with pytest.raises(PreflightError, match='419'):
        check_moved_payload_destinations(certificate_paths=[], strict=True)


def test_unmounted_volume_fails_closed(tmp_path):
    with pytest.raises(PreflightError, match='failed closed'):
        check_moved_payload_destinations(roots=[tmp_path / 'absent'], strict=True)


def test_cli_defines_custody_checks_before_executing():
    import tac.preflight as preflight
    tree = ast.parse(Path(preflight.__file__).read_text())
    definitions = {node.name: i for i, node in enumerate(tree.body)
                   if isinstance(node, ast.FunctionDef)}
    entries = [i for i, node in enumerate(tree.body) if isinstance(node, ast.If)
               and '__name__' in ast.unparse(node.test)]
    assert entries
    for name in ('check_no_bare_cross_arm_artifact_reads', 'check_moved_payload_destinations'):
        assert definitions[name] < min(entries)
