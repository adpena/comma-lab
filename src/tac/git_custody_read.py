"""Read SHA-1 Git custody without spawning a process or touching an index.

Restricted to refs, committed blobs, commit times and ancestry. Packed/loose object
bytes are verified against Git object IDs, including both Git pack delta forms.
"""
from __future__ import annotations

import hashlib
import struct
import zlib
from datetime import UTC, datetime
from pathlib import Path


class GitCustodyReader:
    def __init__(self, repo: Path):
        git = repo / ".git"
        if git.is_file():
            text = git.read_text().strip()
            if not text.startswith("gitdir: "):
                raise ValueError("invalid gitdir")
            git = (repo / text[8:]).resolve()
        self.git = git
        common = git / "commondir"
        self.common = (git / common.read_text().strip()).resolve() if common.exists() else git
        self.objects = self.common / "objects"
        self.cache = {}
        self.pack_cache = {}
        self.indexes = []
        for index in sorted((self.objects / "pack").glob("*.idx")):
            data = index.read_bytes()
            if data[:8] != b"\xfftOc\x00\x00\x00\x02":
                raise ValueError("unsupported Git pack index version")
            count = struct.unpack_from(">I", data, 8 + 255 * 4)[0]
            self.indexes.append((index.with_suffix(".pack"), data, count))

    def resolve(self, ref: str, depth: int = 0) -> str:
        if depth > 8:
            raise ValueError("symbolic ref cycle")
        if len(ref) == 40 and all(c in "0123456789abcdef" for c in ref):
            return ref
        if ref not in {"HEAD", "main"} and not ref.startswith("refs/"):
            raise ValueError("unsupported Git ref")
        path = self.git / ref if ref == "HEAD" else self.common / ("refs/heads/main" if ref == "main" else ref)
        if ".." in Path(ref).parts:
            raise ValueError("invalid ref path")
        if path.exists():
            value = path.read_text().strip()
            return self.resolve(value[5:] if value.startswith("ref: ") else value, depth + 1)
        packed = self.common / "packed-refs"
        if packed.exists():
            wanted = "refs/heads/main" if ref == "main" else ref
            for line in packed.read_text().splitlines():
                if line and not line.startswith(("#", "^")):
                    oid, name = line.split(" ", 1)
                    if name == wanted:
                        return self.resolve(oid, depth + 1)
        raise ValueError(f"unknown Git ref: {ref}")

    def object(self, oid: str) -> tuple[str, bytes]:
        oid = self.resolve(oid)
        if oid in self.cache:
            return self.cache[oid]
        loose = self.objects / oid[:2] / oid[2:]
        if loose.exists():
            encoded = zlib.decompress(loose.read_bytes())
            header, body = encoded.split(b"\0", 1)
            kind, length = header.decode().split(" ")
            if len(body) != int(length):
                raise ValueError("loose object length mismatch")
        else:
            wanted = bytes.fromhex(oid)
            found = None
            for pack, index, count in self.indexes:
                low, high = 0, count
                while low < high:
                    middle = (low + high) // 2
                    item = index[1032 + 20 * middle:1032 + 20 * (middle + 1)]
                    if item < wanted:
                        low = middle + 1
                    else:
                        high = middle
                if low < count and index[1032 + 20 * low:1032 + 20 * (low + 1)] == wanted:
                    offset = struct.unpack_from(">I", index, 1032 + 24 * count + 4 * low)[0]
                    if offset & 0x80000000:
                        offset = struct.unpack_from(">Q", index, 1032 + 28 * count + 8 * (offset & 0x7fffffff))[0]
                    found = self.packed_object(pack, offset)
                    break
            if found is None:
                raise ValueError(f"Git object missing: {oid}")
            kind, body = found
        actual = hashlib.sha1(f"{kind} {len(body)}\0".encode() + body).hexdigest()
        if actual != oid:
            raise ValueError("Git object hash mismatch")
        self.cache[oid] = (kind, body)
        return kind, body

    def packed_object(self, pack: Path, offset: int, depth: int = 0) -> tuple[str, bytes]:
        if depth > 128:
            raise ValueError("Git delta chain too deep")
        if (pack, offset) in self.pack_cache:
            return self.pack_cache[(pack, offset)]
        with pack.open("rb") as stream:
            stream.seek(offset)
            first = stream.read(1)[0]
            kind, size, shift, byte = (first >> 4) & 7, first & 15, 4, first
            while byte & 128:
                byte = stream.read(1)[0]
                size |= (byte & 127) << shift
                shift += 7
                if shift > 63:
                    raise ValueError("invalid pack size")
            base = None
            if kind == 6:
                byte = stream.read(1)[0]
                distance = byte & 127
                while byte & 128:
                    byte = stream.read(1)[0]
                    distance = ((distance + 1) << 7) | (byte & 127)
                if not 0 < distance <= offset:
                    raise ValueError("invalid OFS delta")
                base = (pack, offset - distance)
            elif kind == 7:
                base = stream.read(20).hex()
            decoder = zlib.decompressobj()
            body = bytearray()
            while not decoder.eof:
                chunk = stream.read(65536)
                if not chunk or size > 64 * 1024 * 1024:
                    raise ValueError("truncated or oversized custody object")
                body.extend(decoder.decompress(chunk))
                if len(body) > size:
                    raise ValueError("pack object expands beyond recorded size")
            if len(body) != size:
                raise ValueError("pack length mismatch")
        if kind in (6, 7):
            base_kind, base_body = self.packed_object(*base, depth + 1) if kind == 6 else self.object(base)
            result = (base_kind, apply_git_delta(base_body, bytes(body)))
        else:
            result = ({1: "commit", 2: "tree", 3: "blob", 4: "tag"}[kind], bytes(body))
        self.pack_cache[(pack, offset)] = result
        return result

    def commit(self, ref: str) -> bytes:
        kind, body = self.object(self.resolve(ref))
        if kind != "commit":
            raise ValueError("commit object required")
        return body

    def blob(self, ref: str, path: str) -> bytes:
        parts = Path(path).parts
        if not parts or Path(path).is_absolute() or ".." in parts:
            raise ValueError("invalid committed path")
        tree = self.commit(ref).splitlines()[0].split()[1].decode()
        for index, part in enumerate(parts):
            kind, data = self.object(tree)
            if kind != "tree":
                raise ValueError("tree expected")
            at, child = 0, None
            while at < len(data):
                stop = data.index(b"\0", at)
                _, name = data[at:stop].split(b" ", 1)
                oid = data[stop + 1:stop + 21].hex()
                if name == part.encode():
                    child = oid
                at = stop + 21
            if child is None:
                raise ValueError(f"committed path missing: {path}")
            tree = child
            if index == len(parts) - 1:
                kind, result = self.object(child)
                if kind != "blob":
                    raise ValueError("blob expected")
                return result
        raise ValueError("empty committed path")

    def ancestor(self, older: str, newer: str) -> bool:
        older = self.resolve(older)
        pending, seen = [self.resolve(newer)], set()
        while pending:
            oid = pending.pop()
            if oid == older:
                return True
            if oid in seen:
                continue
            seen.add(oid)
            for line in self.commit(oid).split(b"\n\n", 1)[0].splitlines():
                if line.startswith(b"parent "):
                    pending.append(line[7:].decode())
        return False

    def committed_at(self, ref: str) -> str:
        for line in self.commit(ref).split(b"\n\n", 1)[0].splitlines():
            if line.startswith(b"committer "):
                timestamp = int(line.rsplit(b" ", 2)[1])
                return datetime.fromtimestamp(timestamp, UTC).isoformat()
        raise ValueError("commit timestamp absent")


def apply_git_delta(base: bytes, delta: bytes) -> bytes:
    at = 0
    def varint():
        nonlocal at
        result, shift = 0, 0
        while True:
            byte = delta[at]
            at += 1
            result |= (byte & 127) << shift
            if not byte & 128:
                return result
            shift += 7
            if shift > 63:
                raise ValueError("invalid delta varint")
    if varint() != len(base):
        raise ValueError("delta base length mismatch")
    expected, output = varint(), bytearray()
    while at < len(delta):
        opcode = delta[at]
        at += 1
        if opcode & 128:
            offset = length = 0
            for bit in range(7):
                if opcode & (1 << bit):
                    value = delta[at]
                    at += 1
                    if bit < 4:
                        offset |= value << (8 * bit)
                    else:
                        length |= value << (8 * (bit - 4))
            length = length or 65536
            if offset + length > len(base):
                raise ValueError("delta copy exceeds base")
            output.extend(base[offset:offset + length])
        elif opcode:
            output.extend(delta[at:at + opcode])
            at += opcode
        else:
            raise ValueError("zero delta opcode")
        if len(output) > expected or expected > 64 * 1024 * 1024:
            raise ValueError("delta output exceeds expected size")
    if len(output) != expected or at != len(delta):
        raise ValueError("delta result length mismatch")
    return bytes(output)
