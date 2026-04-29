# Stage 3 Designs — Archive Diet Pass

Scope: CPU-only archive diet for already-trained renderer / SegMap submissions. All score deltas below are rate-only estimates tagged **[advisory only]** because they are computed from archive bytes, not a fresh contest-CUDA authoritative eval.

## Design 1 — High Priority: `archive_diet_pack.py`

1. File path

`src/tac/archive_diet_pack.py`

2. Function signatures

```python
def diet_pack(
    input_archive: Path,
    output_archive: Path,
    *,
    brotli_quality: int = 11,
    verify: bool = True,
) -> dict:
    ...

def _read_zip_members(input_archive: Path) -> dict[str, bytes]: ...
def _write_deterministic_brotli_zip(output_archive: Path, members: dict[str, bytes], quality: int) -> dict[str, int]: ...
def _repack_segmap_weights_to_payload(data: bytes, tmpdir: Path, verify: bool) -> tuple[bytes, bool, dict]: ...
def _verify_output_archive(input_members: dict[str, bytes], output_archive: Path, logical_replacements: dict[str, bytes]) -> bool: ...
```

Return contract:

```python
{
    "input_bytes": int,
    "output_bytes": int,
    "savings_bytes": int,
    "savings_score_pts": float,  # [advisory only] 25*savings_bytes/ORIGINAL_VIDEO_BYTES
    "bit_exact": bool,
    "components": {name: {"in": int, "out": int}},
}
```

3. Integration points

`src/tac/archive_diet_pack.py` imports:

```python
from tac.arithmetic_qint_codec import repack_payload_tar_xz_to_arithmetic, unpack_arithmetic_payload
from tac.block_fp_codec import unpack_payload_tar_xz
from tac.submission_archive import ORIGINAL_VIDEO_BYTES
```

Wire as an encoder-side utility only:

- Add CLI later under `experiments/diet_archive.py` or `submissions/robust_current/compress_archive.py --diet-pack`.
- For SegMap archives, replace `segmap_weights.tar.xz` with `payload.bin`, then Brotli-store as `payload.bin.br`. Existing `inflate.sh` already performs `.br` decompression before Python dispatch; the archive must dispatch to `PYTHON_INFLATE=segmap_arithmetic`.
- For renderer archives, keep `renderer.bin` logically unchanged and Brotli-store `renderer.bin.br`; existing `.br` decompression makes it visible as `renderer.bin`.
- Do not modify upstream scorer files. Do not call `pack_payload_tar_xz(..., exponents=...)`; that kwarg does not exist.

4. Test cases

Positive:

- Renderer archive with `renderer.bin`, `grayscale.mkv` or `masks.mkv`, and `optimized_poses.bin` repacks deterministically; decoded `.br` members are byte-identical to input.
- SegMap archive with `segmap_weights.tar.xz`, `grayscale.mkv`, and `optimized_poses.bin` emits `payload.bin.br`, `grayscale.mkv.br`, `optimized_poses.bin.br`; `unpack_payload_tar_xz` and `unpack_arithmetic_payload` produce bit-exact tensor dictionaries.
- Running `diet_pack` twice on the same input produces byte-identical output archives.
- Returned `savings_score_pts` equals `25 * savings_bytes / ORIGINAL_VIDEO_BYTES` and is documented as [advisory only].

Negative:

- Non-ZIP input raises `ValueError("not a valid zip archive")`.
- Archive with both missing `renderer.bin` and missing `segmap_weights.tar.xz` raises `ValueError("unsupported archive layout")`.
- Corrupt `segmap_weights.tar.xz` raises a codec/tar error before writing a successful result.
- Corrupt Brotli output during verify raises `RuntimeError`.

Edge:

- Empty ZIP raises `ValueError("empty archive")`.
- Existing malformed `payload.bin` or SHv1/AQv1 version mismatch is rejected in verification.
- `brotli_quality < 0` or `> 11` raises `ValueError`.
- Already `.br` members are rejected unless explicitly supported later; double-Brotli is too easy to ship wrong.

5. Implementation skeleton

```python
def diet_pack(input_archive: Path, output_archive: Path, *, brotli_quality: int = 11, verify: bool = True) -> dict:
    input_archive, output_archive = Path(input_archive), Path(output_archive)
    if not 0 <= brotli_quality <= 11: raise ValueError("brotli_quality must be 0..11")
    members = _read_zip_members(input_archive)
    out_members, components, replacements, bit_exact = {}, {}, {}, not verify
    if "segmap_weights.tar.xz" in members:
        payload, ok, stats = _repack_segmap_weights_to_payload(members["segmap_weights.tar.xz"], verify)
        out_members["payload.bin"] = payload; replacements["payload.bin"] = payload; bit_exact = ok
    elif "renderer.bin" in members:
        out_members["renderer.bin"] = members["renderer.bin"]
    else:
        raise ValueError("unsupported archive layout")
    for name, data in members.items():
        if name != "segmap_weights.tar.xz": out_members.setdefault(name, data)
    br_sizes = _write_deterministic_brotli_zip(output_archive, out_members, brotli_quality)
    if verify: bit_exact = _verify_output_archive(members, output_archive, replacements)
    return _stats(input_archive, output_archive, members, out_members, br_sizes, bit_exact)
```

## Design 2 — `manifest_pack.py`

1. File path

`src/tac/manifest_pack.py`

2. Function signatures

```python
MANIFEST_PACK_MAGIC = b"MPK1"

def pack_manifest_archive(input_archive: Path, output_path: Path, *, brotli_quality: int = 11, verify: bool = True) -> dict: ...
def unpack_manifest_archive(input_path: Path, output_dir: Path, *, verify_crc: bool = True) -> list[Path]: ...
def encode_manifest(entries: list[tuple[str, bytes]]) -> bytes: ...
def decode_manifest(blob: bytes) -> list[tuple[str, bytes]]: ...
def write_varint(value: int) -> bytes: ...
def read_varint(buf: memoryview, offset: int) -> tuple[int, int]: ...
```

3. Integration points

- `submissions/robust_current/inflate.sh`: add a stage-0 branch before ZIP extraction:
  - if archive starts with `MPK1`, call `python -m tac.manifest_pack --unpack archive.mpk archive/`.
  - otherwise keep current ZIP path.
- `src/tac/submission_archive.py`: optional `archive_format="zip" | "manifest"` flag after this is proven.
- Renderer compact path only for first ship: `renderer.bin`, `masks.mkv`/`grayscale.mkv`, `optimized_poses.bin`, optional postfilter artifact. SegMap arithmetic can be enabled after Design 1 stabilizes.

4. Test cases

Positive:

- ZIP → MPK → directory roundtrip preserves names, order, bytes, and CRC32.
- MPK is smaller than deterministic ZIP for tiny multi-member archives where central directory overhead dominates.
- Brotli q=11 output is deterministic across two runs.

Negative:

- Bad magic raises `ValueError("bad MPK1 magic")`.
- Declared entry length past EOF raises `ValueError("truncated manifest archive")`.
- Duplicate entry names rejected.
- CRC mismatch rejected during `unpack_manifest_archive`.

Edge:

- Empty archive rejected.
- Version mismatch (`MPK2` or version byte > supported) rejected.
- Varint overflow rejected after 10 bytes.
- Path traversal names (`../x`, `/abs`) rejected.

5. Implementation skeleton

```python
def encode_manifest(entries: list[tuple[str, bytes]]) -> bytes:
    out = bytearray(MANIFEST_PACK_MAGIC); out += write_varint(1); out += write_varint(len(entries))
    for name, payload in entries:
        _validate_name(name)
        name_b = name.encode("utf-8")
        out += write_varint(len(name_b)); out += name_b
        out += write_varint(len(payload)); out += zlib.crc32(payload).to_bytes(4, "little")
    for _, payload in entries:
        out += payload
    return bytes(out)

def pack_manifest_archive(input_archive: Path, output_path: Path, *, brotli_quality: int = 11, verify: bool = True) -> dict:
    entries = _read_zip_entries(input_archive)
    raw = encode_manifest(entries)
    output_path.write_bytes(brotli.compress(raw, quality=brotli_quality, lgwin=24))
    if verify: _assert_entries_equal(entries, decode_manifest(brotli.decompress(output_path.read_bytes())))
    return _stats(input_archive, output_path, entries)
```

## Design 3 — `pose_residual_codec.py`

1. File path

`src/tac/pose_residual_codec.py`

2. Function signatures

```python
POSE_RESIDUAL_MAGIC = b"PRC1"

def encode_pose_residuals(
    poses: torch.Tensor,
    *,
    pose_dim: int = 6,
    delta_bits: int = 8,
    arithmetic: bool = True,
) -> bytes: ...

def decode_pose_residuals(blob: bytes, *, pose_dim: int = 6) -> torch.Tensor: ...
def encode_pose_file(src_path: Path, dst_path: Path, *, pose_dim: int = 6, verify: bool = True) -> dict: ...
def is_pose_residual_blob(blob: bytes) -> bool: ...
```

3. Integration points

- `src/tac/submission_archive.py::load_optimized_poses`: add a raw bytes branch before raw fp16 fallback:
  - if `raw.startswith(b"PRC1")`, call `decode_pose_residuals(raw, pose_dim=pose_dim)`.
- `src/tac/submission_archive.py::build_submission_archive`: no manifest change if shipped as `optimized_poses.bin`; the loader detects by magic.
- `submissions/robust_current/compress_archive.py`: compact path can call `encode_pose_file(..., optimized_poses.bin)` instead of `save_poses_binary` when `--pose-residual-codec` is set.
- This supersedes the torch-pickle `pose_delta_codec.py` for shipping because it removes pickle/container overhead and arithmetic-codes the int8 residual stream.

4. Test cases

Positive:

- Smooth `(600, 6)` pose tensor roundtrips within the same quantization bound as `pose_delta_codec`.
- Arithmetic-coded residual stream is smaller than raw fp16 for smooth synthetic trajectory.
- `load_optimized_poses` transparently decodes `optimized_poses.bin` PRC1 blobs.
- Deterministic encode: identical tensor produces identical bytes.

Negative:

- Bad magic raises `ValueError("bad PRC1 magic")`.
- Version mismatch raises `ValueError("unsupported PRC1 version")`.
- Wrong declared `pose_dim` raises `ValueError`.
- Truncated AQv1 residual payload propagates `ValueError`.

Edge:

- Single-pair pose tensor stores only anchor and zero residual payload.
- All-zero residuals roundtrip and compress aggressively.
- Saturated int8 residual ratio > threshold is reported in stats, not silently ignored.
- Non-2D tensor rejected.

5. Implementation skeleton

```python
def encode_pose_residuals(poses: torch.Tensor, *, pose_dim: int = 6, delta_bits: int = 8, arithmetic: bool = True) -> bytes:
    poses_f = _validate_poses(poses, pose_dim)
    anchor = poses_f[0].to(torch.float16).numpy().tobytes()
    residuals = poses_f[1:] - poses_f[:-1]
    scale = residuals.abs().amax(dim=0).clamp(min=1e-8).to(torch.float16)
    q = ((residuals / scale.float()) * 127).round().clamp(-127, 127).to(torch.int8).cpu().numpy()
    payload = encode_qints_arithmetic(q, num_symbols=255, offset=127) if arithmetic else q.tobytes()
    return b"PRC1" + struct.pack("<HHH", 1, poses_f.shape[0], pose_dim) + scale.numpy().tobytes() + anchor + struct.pack("<I", len(payload)) + payload

def decode_pose_residuals(blob: bytes, *, pose_dim: int = 6) -> torch.Tensor:
    version, n_pairs, declared_dim = _parse_header(blob)
    if version != 1 or declared_dim != pose_dim: raise ValueError(...)
    anchor, scale, payload = _read_body(blob, n_pairs, pose_dim)
    q = decode_qints_arithmetic(payload, expected_dtype=np.int8).reshape(n_pairs - 1, pose_dim)
    deltas = torch.from_numpy(q).float() / 127.0 * scale
    return anchor + torch.cat([torch.zeros(1, pose_dim), torch.cumsum(deltas, dim=0)])
```

## Shipping Order

1. Land Design 1 first. It is lossless, CPU-only, and stackable on any current archive. Rate savings are [advisory only] until the resulting archive is contest-CUDA evaluated.
2. Add Design 3 next if pose bytes still matter after Brotli/SHv1. It is small but requires loader integration.
3. Add Design 2 last because it changes the outer archive contract and needs inflate-stage plumbing, even though the implementation is simple.

Explicit non-goals for this pass:

- No Modal dispatch.
- No trained-checkpoint-dependent Hessian allocation.
- No KL distill.
- No adaptive rebalance.
- No upstream scorer edits.
- No changes to `noise_std`; any path touched later must keep default `0.5`.
- Anchor masks remain `experiments/results/lane_a_landed/iter_0/` for lanes that need them.
