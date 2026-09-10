"""Resolve rp1 MOVED certificates without changing logical artifact identities."""
import hashlib
import json
import os
import shutil
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
    before, digest = _identity(src), _sha(src)
    with src.open('rb') as source, dst.open('xb') as target:
        shutil.copyfileobj(source, target, 1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    if _identity(src) != before or _sha(src) != digest or dst.stat().st_size != before[3] or _sha(dst) != digest:
        raise MovedArtifactError('COPY_OR_SOURCE_DRIFT')
    row = {'moved_to': str(dst), 'sha256': digest, 'bytes': before[3],
           'rebuildable_from': rebuildable_from, 'reason': reason}
    atomic_write_json(manifest, row)
    if _identity(src) != before:
        raise MovedArtifactError('SOURCE_DRIFT_BEFORE_UNLINK')
    src.unlink()
    return dst
