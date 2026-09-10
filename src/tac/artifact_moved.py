"""Resolve rp1 MOVED certificates without changing logical artifact identities."""
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

from tac.micro_edit.ledger import atomic_write_json

_VERIFIED: dict[tuple, str] = {}
class MovedArtifactError(ValueError):
    """Missing, malformed, cyclic, or drifted artifact custody; never permission to delete."""
def _identity(path):
    s = path.stat()
    return (str(path), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
def _sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
def verify_payload(path, expected_bytes, expected_sha256, *, hash_payload=True):
    """Re-read a regular data fork; metadata sidecars cannot certify payload custody.

    Small legitimate payloads are allowed: AppleDouble is identified by name or
    magic, not an arbitrary minimum size. No cache is used for certification.
    """
    path = Path(path)
    if (type(expected_bytes) is not int or expected_bytes < 0
            or not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise MovedArtifactError('INVALID_PAYLOAD_IDENTITY')
    if any(part.startswith('._') for part in path.parts):
        raise MovedArtifactError(f'APPLEDOUBLE_NAME:{path}')
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise MovedArtifactError(f'NOT_REGULAR_DATA_FORK:{path}')
    before = _identity(path)
    with path.open('rb') as stream:
        if stream.read(4) == b'\x00\x05\x16\x07':
            raise MovedArtifactError(f'APPLEDOUBLE_HEADER:{path}')
    if before[3] != expected_bytes:
        raise MovedArtifactError(f'BYTE_DRIFT:{path}')
    digest = _sha(path) if hash_payload else None
    if (before != _identity(path) or path.is_symlink()
            or (hash_payload and digest != expected_sha256)):
        raise MovedArtifactError(f'SHA_OR_IDENTITY_DRIFT:{path}')
    return {'path': str(path), 'bytes': before[3], 'sha256': digest,
            'identity': before, 'hash_verified': hash_payload}


def copy_verified(src, dst, *, expected_bytes=None, expected_sha256=None):
    """Keep source and any interrupted output; never overwrite an existing path."""
    src, dst = Path(src).absolute(), Path(dst).absolute()
    if src.is_symlink() or not src.is_file() or os.path.lexists(dst):
        raise MovedArtifactError('COPY_PRECONDITION_REFUSED')
    before = _identity(src)
    digest = _sha(src) if expected_sha256 is None else expected_sha256
    size = before[3] if expected_bytes is None else expected_bytes
    verify_payload(src, size, digest)
    if any(part.startswith('._') for part in dst.parts):
        raise MovedArtifactError(f'APPLEDOUBLE_NAME:{dst}')
    with src.open('rb') as source, dst.open('xb') as target:
        shutil.copyfileobj(source, target, 1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    result = verify_payload(dst, size, digest)
    if _identity(src) != before:
        raise MovedArtifactError('SOURCE_DRIFT_DURING_COPY')
    return result
def resolve(path, *, receipt=None, max_depth=8) -> Path:
    path, chain, seen = Path(path), [], set()
    try:
        while not path.exists():
            key = str(path.absolute())
            if key in seen or len(chain) >= min(max_depth, 8):
                raise MovedArtifactError(f'CHAIN_LIMIT_OR_CYCLE:{path}')
            seen.add(key)
            manifest = path.with_name(path.name + '.MOVED.json')
            row = json.loads(manifest.read_text())
            if (not isinstance(row, dict) or type(row.get('bytes')) is not int or row['bytes'] < 0
                    or any(not isinstance(row.get(k), str) or not row[k].strip()
                           for k in ('moved_to', 'sha256', 'reason', 'rebuildable_from'))
                    or len(row['sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['sha256'])):
                raise MovedArtifactError(f'INVALID_MANIFEST:{manifest}')
            chain.append((str(manifest), row))
            path = Path(row['moved_to'])
            if not path.is_absolute():
                raise MovedArtifactError(f'NON_ABSOLUTE_DESTINATION:{manifest}')
        if chain:
            before = _identity(path)
            if any(row['bytes'] != before[3] for _, row in chain):
                raise MovedArtifactError(f'BYTE_DRIFT:{path}')
            cached = before in _VERIFIED
            digest = _VERIFIED[before] if cached else _sha(path)
            if before != _identity(path) or any(row['sha256'] != digest for _, row in chain):
                raise MovedArtifactError(f'SHA_OR_IDENTITY_DRIFT:{path}')
            _VERIFIED[before] = digest
            if receipt is not None:
                receipt.append({"path": str(path), "bytes": before[3], "sha256": digest,
                                "identity": before, "cached": cached, "manifests": [m for m, _ in chain]})
        return path
    except (OSError, TypeError, KeyError, ValueError) as exc:
        raise MovedArtifactError(f'RESOLUTION_REFUSED:{path}:{exc}') from exc
def move_with_manifest(src, dst, reason, rebuildable_from) -> Path:
    """Quiescent files only: verified copy, atomic rp1 certificate, then unlink; no overwrite."""
    src, dst = Path(src).absolute(), Path(dst).absolute()
    manifest = src.with_name(src.name + '.MOVED.json')
    if (src.is_symlink() or not src.is_file() or dst.exists() or manifest.exists()
            or any(not isinstance(v, str) or not v.strip() for v in (reason, rebuildable_from))):
        raise MovedArtifactError('MOVE_PRECONDITION_REFUSED')
    before = _identity(src)
    copied = copy_verified(src, dst)
    digest = copied['sha256']
    if _identity(src) != before or tuple(copied['identity']) != _identity(dst):
        raise MovedArtifactError('COPY_OR_SOURCE_DRIFT')
    row = {'moved_to': str(dst), 'sha256': digest, 'bytes': before[3],
           'rebuildable_from': rebuildable_from, 'reason': reason}
    atomic_write_json(manifest, row)
    if _identity(src) != before or tuple(copied['identity']) != _identity(dst):
        raise MovedArtifactError('SOURCE_DRIFT_BEFORE_UNLINK')
    src.unlink()
    return dst
