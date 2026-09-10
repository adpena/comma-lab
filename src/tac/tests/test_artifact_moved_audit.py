"""Behavioral custody fixtures: never scan real volume roots."""
import hashlib
import json

import pytest

from tac import artifact_moved_audit as audit


@pytest.fixture
def certificate(tmp_path):
    payload = tmp_path / "retained.raw"
    data = b"real retained payload\n" * 16
    payload.write_bytes(data)
    row = {"moved_to": str(payload), "bytes": len(data),
           "sha256": hashlib.sha256(data).hexdigest(), "reason": "cold store",
           "rebuildable_from": "retained source"}
    manifest = tmp_path / "source.raw.MOVED.json"
    manifest.write_text(json.dumps(row))
    return manifest, payload, row


def log_file(tmp_path, rows):
    path = tmp_path / "MOVE_LOG.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_valid_destination_without_source(certificate):
    manifest, _, _ = certificate
    assert audit.certificate_violations([manifest], hash_payloads=True) == []


def test_missing_destination(certificate):
    manifest, payload, _ = certificate
    payload.unlink()
    assert audit.certificate_violations([manifest])


def test_stub_sibling_does_not_replace_missing_payload(certificate):
    manifest, payload, _ = certificate
    payload.unlink()
    payload.with_name("._" + payload.name).write_bytes(b"metadata")
    assert audit.certificate_violations([manifest])


def test_appledouble_header(certificate):
    manifest, payload, row = certificate
    payload.write_bytes(b"\x00\x05\x16\x07" + b"x" * (row["bytes"] - 4))
    assert audit.certificate_violations([manifest])


def test_appledouble_named_destination(certificate):
    manifest, payload, row = certificate
    stub = payload.with_name("._" + payload.name)
    payload.rename(stub)
    row["moved_to"] = str(stub)
    manifest.write_text(json.dumps(row))
    assert audit.certificate_violations([manifest])


def test_size_mismatch(certificate):
    manifest, payload, _ = certificate
    payload.write_bytes(b"short")
    assert audit.certificate_violations([manifest])


def test_same_size_corruption_requires_hash_mode(certificate):
    manifest, payload, row = certificate
    payload.write_bytes(b"z" * row["bytes"])
    assert audit.certificate_violations([manifest]) == []
    assert audit.certificate_violations([manifest], hash_payloads=True)


def test_default_passes_explicit_no_hash_to_verifier(certificate, monkeypatch):
    calls = []
    monkeypatch.setattr(audit, "verify_payload", lambda *args, **kwargs: calls.append(kwargs))
    assert audit.certificate_violations([certificate[0]]) == []
    assert calls == [{"hash_payload": False}]


@pytest.mark.parametrize("change", [{"bytes": True}, {"bytes": -1}, {"sha256": "bad"},
                                     {"moved_to": "relative.raw"}])
def test_bad_identity(certificate, change):
    manifest, _, row = certificate
    manifest.write_text(json.dumps(dict(row, **change)))
    assert audit.certificate_violations([manifest])


def test_directory_destination(certificate):
    manifest, payload, row = certificate
    row["moved_to"] = str(payload.parent)
    manifest.write_text(json.dumps(row))
    assert audit.certificate_violations([manifest])


def test_malformed_json_and_nonobjects(tmp_path):
    path = tmp_path / "MOVED.json"
    for text in ("{", "[]", '{"bytes":1,"bytes":2}'):
        path.write_text(text)
        assert audit.certificate_violations([path])


def test_legacy_directory_schema_explicitly_refused(tmp_path):
    path = tmp_path / "MOVED.json"
    path.write_text(json.dumps({"source_root": "/original", "destination_root": "/retained", "files": []}))
    assert "UNSUPPORTED_DIRECTORY_SCHEMA" in audit.certificate_violations([path])[0]


def test_legacy_path_log_uses_exact_sibling_manifest(certificate, tmp_path):
    manifest, _, row = certificate
    log = log_file(tmp_path, [{"status": "MOVED", "path": str(manifest)[:-11],
                               "bytes": row["bytes"], "sha256": row["sha256"]}])
    assert audit.certificate_violations([log], hash_payloads=True) == []


def test_legacy_log_without_manifest_does_not_guess(tmp_path):
    log = log_file(tmp_path, [{"status": "MOVED", "path": str(tmp_path / "absent.raw")}])
    assert audit.certificate_violations([log])


def test_legacy_log_identity_mismatch(certificate, tmp_path):
    manifest, _, _ = certificate
    log = log_file(tmp_path, [{"status": "MOVED", "path": str(manifest)[:-11], "bytes": 99}])
    assert "IDENTITY_MISMATCH" in audit.certificate_violations([log])[0]


def test_correction_does_not_waive_missing_bytes(certificate, tmp_path):
    manifest, payload, row = certificate
    payload.unlink()
    source = str(manifest)[:-11]
    log = log_file(tmp_path, [dict(row, status="MOVED", path=source),
                              {"status": "MOVED_CERTIFICATE_FALSE", "path": source}])
    assert audit.certificate_violations([log])


def test_restored_destination_and_recovery_pass(certificate, tmp_path):
    manifest, _, row = certificate
    source = str(manifest)[:-11]
    log = log_file(tmp_path, [dict(row, status="MOVED", path=source),
                              {"status": "MOVED_CERTIFICATE_FALSE", "path": source},
                              dict(row, status="RECOVERED", path=source)])
    assert audit.certificate_violations([log], hash_payloads=True) == []


def test_recovery_row_is_independently_checked(certificate, tmp_path):
    _, payload, row = certificate
    payload.unlink()
    log = log_file(tmp_path, [dict(row, status="RECOVERED")])
    assert audit.certificate_violations([log])


def test_empty_and_malformed_logs_are_failures(tmp_path):
    log = tmp_path / "MOVE_LOG.jsonl"
    for text in ("", "\n", "{\n", '{}\n', '{"status": []}\n'):
        log.write_text(text)
        assert audit.certificate_violations([log])


def test_one_bad_row_does_not_hide_later_failure(certificate, tmp_path):
    _, payload, row = certificate
    payload.unlink()
    log = log_file(tmp_path, [{"status": "unknown"}, dict(row, status="MOVED")])
    failures = audit.certificate_violations([log])
    assert len(failures) == 2
    assert ":1:" in failures[0] and ":2:" in failures[1]


def test_missing_input_and_stub_certificate_are_failures(tmp_path):
    path = tmp_path / "missing.MOVED.json"
    stub = tmp_path / "._MOVED.json"
    stub.write_bytes(b"metadata")
    assert len(audit.certificate_violations([path, stub])) == 2


def test_recovery_elsewhere_does_not_waive_historical_destination(certificate, tmp_path):
    _, payload, row = certificate
    recovered = tmp_path / "recovered.raw"
    payload.rename(recovered)
    log = log_file(tmp_path, [dict(row, status="MOVED"),
                              dict(row, status="RECOVERED", moved_to=str(recovered))])
    failures = audit.certificate_violations([log], hash_payloads=True)
    assert len(failures) == 1 and ":1:" in failures[0]


def test_discovery_exact_names_and_depth(tmp_path):
    for name in ("MOVED.json", "file.MOVED.json", "MOVE_LOG.jsonl", "._file.MOVED.json", "xMOVED.json"):
        (tmp_path / name).write_text("{}")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "child.MOVED.json").write_text("{}")
    assert audit.discover_certificates([tmp_path], max_depth=1) == sorted([
        tmp_path / "file.MOVED.json", tmp_path / "MOVE_LOG.jsonl", tmp_path / "MOVED.json"])
    assert nested / "child.MOVED.json" in audit.discover_certificates([tmp_path], max_depth=2)


def test_discovery_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit.discover_certificates([tmp_path / "missing"])


def test_discovery_read_error_raises(tmp_path, monkeypatch):
    def refused(path):
        raise PermissionError(str(path))
    monkeypatch.setattr(audit.os, "scandir", refused)
    with pytest.raises(PermissionError):
        audit.discover_certificates([tmp_path])


def test_discovery_does_not_follow_symlink_directories(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.MOVED.json").write_text("{}")
    (root / "alias").symlink_to(outside, target_is_directory=True)
    (root / "cycle").symlink_to(root, target_is_directory=True)
    assert audit.discover_certificates([root]) == []
    with pytest.raises(ValueError, match="ROOT_NOT_REAL_DIRECTORY"):
        audit.discover_certificates([root / "alias"])


def test_discovery_symlink_certificate_raises(tmp_path):
    (tmp_path / "file.MOVED.json").symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="CERTIFICATE_NOT_REGULAR"):
        audit.discover_certificates([tmp_path])


def test_discovery_invalid_scope_raises(tmp_path):
    for depth in (0, -1, True):
        with pytest.raises(ValueError, match="MAX_DEPTH"):
            audit.discover_certificates([tmp_path], max_depth=depth)
    with pytest.raises(ValueError, match="EMPTY_DISCOVERY_ROOTS"):
        audit.discover_certificates([])


def test_discovery_overlapping_roots_deduplicate(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    path = nested / "file.MOVED.json"
    path.write_text("{}")
    assert audit.discover_certificates([tmp_path, nested]) == [path]


def test_discovery_bare_file_certificate_is_checked(certificate):
    manifest, payload, row = certificate
    bare = manifest.parent / "MOVED.json"
    bare.write_text(json.dumps(row))
    assert bare in audit.discover_certificates([manifest.parent])
    payload.unlink()
    assert audit.certificate_violations([bare])


@pytest.mark.parametrize("text", ["{", "{}", "[]", '{"schema":"unknown"}',
                                   '{"schema":"vertigo_cold_move.v2"}'])
def test_discovery_unknown_or_malformed_bare_remains_visible(tmp_path, text):
    bare = tmp_path / "MOVED.json"
    bare.write_text(text)
    assert audit.discover_certificates([tmp_path]) == [bare]
    assert audit.certificate_violations([bare])


@pytest.mark.parametrize("schema", ["vertigo_cold_move.v2", "ddm_sr2_vertigo_move_v1"])
def test_discovery_only_known_directory_schema_is_excluded(tmp_path, schema):
    bare = tmp_path / "MOVED.json"
    row = {"schema": schema, "original_path": str(tmp_path / "source"),
           "destination": str(tmp_path / "cold")}
    bare.write_text(json.dumps(row))
    assert audit.discover_certificates([tmp_path]) == []
    bare.write_text(json.dumps(dict(row, moved_to=str(tmp_path / "payload"))))
    assert audit.discover_certificates([tmp_path]) == [bare]
    bare.write_text(json.dumps(dict(row, destination="relative")))
    assert audit.discover_certificates([tmp_path]) == [bare]
