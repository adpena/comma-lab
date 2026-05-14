# SPDX-License-Identifier: MIT
"""Archive-diet pipeline (2026-04-29).

Reduces submission archive byte count via lossless / near-lossless
re-encodings of the inner members. Operates on an existing archive.zip
(typically the Lane G v3 anchor) and writes a smaller sibling archive
that produces equivalent inflate-time outputs.

Available techniques (CLI flag --techniques value, comma-joined):

* ``pose_delta`` — re-encode ``optimized_poses.pt`` via Lane PD
  (``tac.pose_delta_codec``). Lossless to ~0.5% per pose-dim.
  Verified savings on Lane PD design doc: 7200 B → ~3700 B (-49%).

* ``zstd_renderer`` — recompress the renderer.bin payload bytes inside
  the zip with zstandard level 22 (a strictly heavier coder than the
  zip's default DEFLATE compresslevel=9). The renderer.bin contents
  themselves are unchanged — only the *outer zip member compression*
  changes by re-emitting that member with stored bytes wrapped in a
  zstd frame and a tiny ``ZDR1`` header. **Note:** because contest
  inflate.sh expects the archive to be a plain zip, this technique
  is reserved for future Lane DT-V2 where inflate.sh learns the
  ``ZDR1`` header. For Lane G v3 baseline, this technique no-ops.

* ``zip_recompress`` — re-emit each member with ``compresslevel=9``
  but force LZMA compression (``ZIP_LZMA``) instead of DEFLATE. Some
  zip readers in inflate.sh tolerate this; verify before shipping.

* ``arithmetic_renderer`` — if renderer.bin starts with a known
  Selfcomp tar.xz payload magic (``"\\xfd7zXZ\\x00"``), repack to the
  Lane SH ``SHv1`` format using
  :func:`tac.arithmetic_qint_codec.repack_payload_tar_xz_to_arithmetic`.
  No-op for ASYM/FP4A/etc. magics (which use opaque per-architecture
  containers, not tar.xz).

* ``mkv_passthrough`` — store ``masks.mkv`` with ``ZIP_STORED``
  (no DEFLATE attempt). AV1 is incompressible; the default DEFLATE
  pass on a 411 KB AV1 file wastes CPU and may even add bytes.

The diet pipeline preserves member ordering and emits a deterministic
zip (fixed timestamp, fixed compresslevel) so the same input archive
produces the same output bytes on every host.

Public API
----------
* :func:`diet_archive` — input zip + technique list -> output zip + stats.
* :func:`verify_diet_archive` — read both archives, decode their pose
  / renderer / mask members, return correctness + max delta.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

# Fixed zip timestamp matches submission_archive / stack_compositions for
# byte-identical output across hosts.
_FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

# tar.xz magic prefix (xz container header, not the gzip-style magic).
_XZ_MAGIC = b"\xfd7zXZ\x00"

# Order in which we emit members. Matches stack_compositions.REQUIRED_ARCHIVE_MEMBERS
# but we also tolerate archives that omit gradient_corrections.bin.
_CANONICAL_MEMBER_ORDER: tuple[str, ...] = (
    "renderer.bin",
    "masks.mkv",
    "optimized_poses.pt",
    "gradient_corrections.bin",
)


@dataclass
class DietResult:
    """Summary returned by :func:`diet_archive`."""

    input_bytes: int
    output_bytes: int
    savings_bytes: int
    savings_pct: float
    per_member_in: dict[str, int]
    per_member_out: dict[str, int]
    techniques_applied: list[str]
    techniques_noop: list[str]

    def as_dict(self) -> dict:
        return {
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "savings_bytes": self.savings_bytes,
            "savings_pct": self.savings_pct,
            "per_member_in": self.per_member_in,
            "per_member_out": self.per_member_out,
            "techniques_applied": self.techniques_applied,
            "techniques_noop": self.techniques_noop,
        }


def _read_zip_members(zip_path: Path) -> list[tuple[str, bytes]]:
    """Return [(name, payload_bytes)] in archive order."""
    out: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(zip_path, mode="r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            payload = zf.read(info.filename)
            out.append((info.filename, payload))
    return out


def _write_zip_deterministic(
    zip_path: Path,
    members: list[tuple[str, bytes, int]],
) -> None:
    """Write ``members`` (name, payload, compress_type) deterministically.

    ``compress_type`` is one of ``zipfile.ZIP_DEFLATED`` /
    ``zipfile.ZIP_STORED`` / ``zipfile.ZIP_LZMA``.
    """
    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:
        for name, payload, ctype in members:
            info = zipfile.ZipInfo(filename=name, date_time=_FIXED_ZIP_TIMESTAMP)
            info.compress_type = ctype
            info.create_system = 3  # unix
            if ctype == zipfile.ZIP_DEFLATED:
                zf.writestr(info, payload, compresslevel=9)
            else:
                zf.writestr(info, payload)


def _apply_pose_delta(payload: bytes) -> tuple[bytes, bool]:
    """Re-encode optimized_poses.pt via Lane PD.

    Returns ``(new_payload, applied)``. ``applied=False`` if the input is
    already in Lane PD format or the re-encode would be larger.
    """
    from tac.pose_delta_codec import (
        encode_pose_deltas,
        is_pose_delta_dict,
    )

    obj = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and is_pose_delta_dict(obj):
        return payload, False
    if not isinstance(obj, torch.Tensor):
        # Some archives store optimized_poses.pt as a dict {"poses": tensor};
        # try to extract.
        if isinstance(obj, dict) and "poses" in obj and isinstance(obj["poses"], torch.Tensor):
            poses = obj["poses"]
        else:
            return payload, False
    else:
        poses = obj
    if poses.ndim != 2 or poses.shape[0] < 2:
        return payload, False
    encoded = encode_pose_deltas(poses)
    buf = io.BytesIO()
    torch.save(encoded, buf)
    new_payload = buf.getvalue()
    if len(new_payload) >= len(payload):
        # Lane PD shouldn't lose to absolute fp16 unless the trajectory
        # is non-smooth; but be safe.
        return payload, False
    return new_payload, True


def _apply_arithmetic_renderer(payload: bytes) -> tuple[bytes, bool]:
    """If renderer.bin is a Selfcomp xz payload, repack to Lane SH SHv1.

    No-op for ASYM/FP4A/etc. magics. Returns ``(new_payload, applied)``.
    """
    if not payload.startswith(_XZ_MAGIC):
        return payload, False

    import tempfile
    from tac.arithmetic_qint_codec import repack_payload_tar_xz_to_arithmetic

    with tempfile.NamedTemporaryFile(suffix=".tar.xz", delete=False) as in_f:
        in_path = Path(in_f.name)
        in_f.write(payload)
    out_path = in_path.with_suffix(".bin")
    try:
        repack_payload_tar_xz_to_arithmetic(str(in_path), str(out_path))
        new_payload = out_path.read_bytes()
    finally:
        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
    if len(new_payload) >= len(payload):
        return payload, False
    return new_payload, True


def _resolve_compress_type(member_name: str, techniques: set[str]) -> int:
    """Decide the per-member zip compress_type given the technique set."""
    if member_name == "masks.mkv" and "mkv_passthrough" in techniques:
        return zipfile.ZIP_STORED
    if "zip_recompress" in techniques:
        # LZMA inside zip — heavier than DEFLATE on opaque tensor blobs.
        return zipfile.ZIP_LZMA
    return zipfile.ZIP_DEFLATED


def diet_archive(
    input_zip: Path | str,
    output_zip: Path | str,
    techniques: Iterable[str] = (),
) -> DietResult:
    """Apply the diet techniques to ``input_zip`` and write ``output_zip``.

    Args:
        input_zip: existing submission-style archive containing renderer.bin
            and optimized_poses.pt.
        output_zip: destination for the diet archive.
        techniques: iterable of technique names. Unknown names are ignored
            (with a noop classification).

    Returns:
        :class:`DietResult` summary.
    """
    input_zip = Path(input_zip)
    output_zip = Path(output_zip)
    techniques_set = set(techniques)
    known = {
        "pose_delta",
        "arithmetic_renderer",
        "mkv_passthrough",
        "zip_recompress",
    }
    applied: list[str] = []
    noop: list[str] = []
    for t in techniques_set:
        if t not in known:
            noop.append(t)

    in_members = _read_zip_members(input_zip)
    per_in: dict[str, int] = {n: len(b) for n, b in in_members}
    new_members: list[tuple[str, bytes]] = []
    for name, payload in in_members:
        if name == "optimized_poses.pt" and "pose_delta" in techniques_set:
            new_payload, did = _apply_pose_delta(payload)
            if did and "pose_delta" not in applied:
                applied.append("pose_delta")
            payload = new_payload
        elif name == "renderer.bin" and "arithmetic_renderer" in techniques_set:
            new_payload, did = _apply_arithmetic_renderer(payload)
            if did and "arithmetic_renderer" not in applied:
                applied.append("arithmetic_renderer")
            elif not did and "arithmetic_renderer" not in noop:
                noop.append("arithmetic_renderer")
            payload = new_payload
        new_members.append((name, payload))

    # Determine canonical member ordering (preserves any unknown extras
    # at the end in their original input order).
    ordered: list[tuple[str, bytes, int]] = []
    seen: set[str] = set()
    by_name = {n: p for n, p in new_members}
    for canonical in _CANONICAL_MEMBER_ORDER:
        if canonical in by_name:
            ctype = _resolve_compress_type(canonical, techniques_set)
            ordered.append((canonical, by_name[canonical], ctype))
            seen.add(canonical)
    for n, p in new_members:
        if n not in seen:
            ordered.append((n, p, _resolve_compress_type(n, techniques_set)))

    if "mkv_passthrough" in techniques_set and "masks.mkv" in by_name:
        applied.append("mkv_passthrough")
    if "zip_recompress" in techniques_set:
        applied.append("zip_recompress")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    _write_zip_deterministic(output_zip, ordered)

    out_members = _read_zip_members(output_zip)
    per_out: dict[str, int] = {n: len(b) for n, b in out_members}

    in_bytes = input_zip.stat().st_size
    out_bytes = output_zip.stat().st_size
    return DietResult(
        input_bytes=in_bytes,
        output_bytes=out_bytes,
        savings_bytes=in_bytes - out_bytes,
        savings_pct=(1 - out_bytes / in_bytes) * 100 if in_bytes else 0.0,
        per_member_in=per_in,
        per_member_out=per_out,
        techniques_applied=sorted(set(applied)),
        techniques_noop=sorted(set(noop)),
    )


def verify_diet_archive(
    input_zip: Path | str,
    output_zip: Path | str,
    pose_dim: int = 6,
) -> dict:
    """Verify diet output decodes to equivalent tensors as the input.

    For each member, performs a content-aware comparison:

    * ``optimized_poses.pt`` — load both as tensors (decoding Lane PD if
      present) and check ``max_abs_diff <= 1.0`` (Lane PD spec allows
      ~0.5% per-dim error which can be O(1) on a ~5 m camera position
      delta; the renderer is robust to that).
    * ``renderer.bin`` — bytes are equal (we never lossy-encode renderer
      payloads in the default technique set; arithmetic_renderer is
      bit-exact so the tensors decoded from either side match).
    * ``masks.mkv`` — bytes equal (we only change the outer zip's
      compress_type).

    Returns:
        ``{"ok": bool, "per_member": {name: {"ok", ...}}}``.
    """
    input_zip = Path(input_zip)
    output_zip = Path(output_zip)
    in_members = dict(_read_zip_members(input_zip))
    out_members = dict(_read_zip_members(output_zip))

    per: dict[str, dict] = {}
    overall_ok = True

    common = set(in_members) | set(out_members)
    for name in sorted(common):
        if name not in in_members or name not in out_members:
            per[name] = {"ok": False, "reason": "missing"}
            overall_ok = False
            continue
        a = in_members[name]
        b = out_members[name]
        if name == "optimized_poses.pt":
            try:
                from tac.pose_delta_codec import decode_pose_deltas, is_pose_delta_dict

                def _to_tensor(bts: bytes) -> torch.Tensor:
                    obj = torch.load(io.BytesIO(bts), map_location="cpu", weights_only=False)
                    if isinstance(obj, dict) and is_pose_delta_dict(obj):
                        return decode_pose_deltas(obj, pose_dim=pose_dim)
                    if isinstance(obj, dict) and "poses" in obj:
                        return obj["poses"].float()
                    return obj.float()

                ta = _to_tensor(a)
                tb = _to_tensor(b)
                diff = (ta - tb).abs().max().item() if ta.shape == tb.shape else float("inf")
                ok = diff <= 1.0
                per[name] = {"ok": ok, "max_abs_diff": diff, "shape": list(ta.shape)}
                if not ok:
                    overall_ok = False
            except Exception as exc:  # noqa: BLE001
                per[name] = {"ok": False, "reason": f"decode-error: {exc}"}
                overall_ok = False
        else:
            ok = a == b
            per[name] = {"ok": ok, "in_bytes": len(a), "out_bytes": len(b)}
            if not ok:
                overall_ok = False

    return {"ok": overall_ok, "per_member": per}


__all__ = [
    "DietResult",
    "diet_archive",
    "verify_diet_archive",
]
