"""Git custody byte parity, including both packed delta encodings; no index writes."""
from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

import pytest

from tac.git_custody_read import GitCustodyReader, apply_git_delta


def oid(kind, body):
    return hashlib.sha1(f"{kind} {len(body)}\0".encode() + body).digest()


def header(kind, size):
    result = bytearray([(kind << 4) | (size & 15)])
    size >>= 4
    while size:
        result[-1] |= 128
        result.append(size & 127)
        size >>= 7
    return bytes(result)


def packed_repo(tmp_path, delta_type):
    git = tmp_path / ".git"
    store = git / "objects/pack"
    store.mkdir(parents=True)
    base, target = b"source-line\n", b"changed-line\n"
    delta = bytes([len(base), len(target), len(target)]) + target
    base_entry = header(3, len(base)) + zlib.compress(base)
    target_offset = 12 + len(base_entry)
    base_ref = oid("blob", base) if delta_type == 7 else bytes([len(base_entry)])
    target_entry = header(delta_type, len(delta)) + base_ref + zlib.compress(delta)
    body = b"PACK" + struct.pack(">II", 2, 2) + base_entry + target_entry
    checksum = hashlib.sha1(body).digest()
    pack = store / "pack-fixture.pack"
    pack.write_bytes(body + checksum)
    entries = sorted([(oid("blob", base), 12, base_entry), (oid("blob", target), target_offset, target_entry)])
    fanout = [sum(key[0] <= byte for key, _, _ in entries) for byte in range(256)]
    index = b"\xfftOc" + struct.pack(">I", 2) + struct.pack(">256I", *fanout)
    index += b"".join(key for key, _, _ in entries)
    index += b"".join(struct.pack(">I", zlib.crc32(entry)) for _, _, entry in entries)
    index += b"".join(struct.pack(">I", offset) for _, offset, _ in entries)
    index += checksum
    index += hashlib.sha1(index).digest()
    pack.with_suffix(".idx").write_bytes(index)
    return GitCustodyReader(tmp_path), oid("blob", target).hex(), target


@pytest.mark.parametrize("kind", [6, 7])
def test_packed_delta_forms(tmp_path, monkeypatch, kind):
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("Git custody spawned a process"))
    reader, digest, expected = packed_repo(tmp_path, kind)
    assert reader.object(digest) == ("blob", expected)


def test_loose_objects_hash_checked(tmp_path):
    body = b"committed source\n"
    digest = oid("blob", body).hex()
    path = tmp_path / ".git/objects" / digest[:2] / digest[2:]
    path.parent.mkdir(parents=True)
    path.write_bytes(zlib.compress(f"blob {len(body)}\0".encode() + body))
    assert GitCustodyReader(tmp_path).object(digest) == ("blob", body)
    path.write_bytes(zlib.compress(b"blob 7\0changed"))
    with pytest.raises(ValueError, match="hash mismatch"):
        GitCustodyReader(tmp_path).object(digest)


def test_delta_copy_and_malformed_bounds():
    assert apply_git_delta(b"abcd", b"\x04\x04\x90\x04") == b"abcd"
    with pytest.raises(ValueError, match="exceeds"):
        apply_git_delta(b"abc", b"\x03\x04\x90\x04")


def test_live_committed_custody_without_process(monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("Git custody spawned a process"))
    reader = GitCustodyReader(Path(__file__).resolve().parents[3])
    commit = "d2803c2148b6153e6fc26d18428cd2d3cda3ea3e"
    assert reader.ancestor(commit, "HEAD")
    assert reader.blob(commit, "src/tac/decode_wall_clock.py").startswith(b'"""Measured decode budget')
    assert reader.committed_at(commit).endswith("+00:00")
