"""Archive-as-codebook: tile-based texture codec with scorer correction (Trick 35).

Instead of storing full RGB frames or running a neural renderer, encode
the video as:
    1. A small learnable texture atlas (32 atoms, 16x16 each)
    2. A compact per-frame tile assignment map (which atom goes where)
    3. A motion field (how textures flow between frames)
    4. Per-pixel scorer correction targets (pre-computed adjustments)

Total archive target: ~15KB (well within contest rate budget).

Theory: the scoring formula is dominated by SegNet (100x weight).
SegNet cares about semantic class boundaries, not pixel-exact textures.
A 32-atom codebook can represent the 5 semantic classes with enough
texture variety to fool SegNet, while the correction targets fix any
remaining PoseNet distortion.

The codebook + assignments + corrections are stored as raw bytes in
archive.zip alongside the compressed video, so they count toward rate.

Usage::

    codebook = TextureAtomCodebook(num_atoms=32, atom_size=16)
    motion = MotionFieldCodec()
    corrections = ScorerCorrectionTargets()
    archive = build_minimal_archive(masks, codebook, motion, corrections)
"""

from __future__ import annotations

import io
import struct
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureAtomCodebook(nn.Module):
    """Learnable set of texture patches that tile to create frame regions.

    Each atom is a small RGB patch (default 16x16) that represents a
    common texture in the driving video (road surface, sky gradient,
    car body, lane marking, vegetation).  Atoms are learned end-to-end
    by backpropagating through the scorer.

    The codebook is stored as FP16 in the archive for space efficiency.
    32 atoms * 16*16*3 * 2 bytes = 49,152 bytes (~48KB at FP16).
    With 4-bit quantization: 32 * 16*16*3 / 2 = ~12KB.

    Args:
        num_atoms: number of texture atoms in the codebook.
        atom_size: spatial size of each atom (square patch).
        num_channels: color channels (3 for RGB).
    """

    def __init__(
        self,
        num_atoms: int = 32,
        atom_size: int = 16,
        num_channels: int = 3,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.atom_size = atom_size
        self.num_channels = num_channels

        # Learnable atoms initialized as uniform gray with small noise
        self.atoms = nn.Parameter(
            torch.full((num_atoms, num_channels, atom_size, atom_size), 128.0)
            + torch.randn(num_atoms, num_channels, atom_size, atom_size) * 5.0
        )

    def forward(self, assignment_map: torch.Tensor) -> torch.Tensor:
        """Render a frame by tiling atoms according to the assignment map.

        Args:
            assignment_map: (B, H_tiles, W_tiles) long tensor with atom
                indices in [0, num_atoms).  H_tiles = H // atom_size,
                W_tiles = W // atom_size.

        Returns:
            (B, 3, H, W) rendered frame in [0, 255] where
            H = H_tiles * atom_size, W = W_tiles * atom_size.
        """
        B, Ht, Wt = assignment_map.shape
        device = assignment_map.device
        A = self.atom_size
        C = self.num_channels

        # Clamp atoms to valid range
        atoms_clamped = self.atoms.clamp(0.0, 255.0)

        # Gather atoms for each tile position
        # (B, Ht, Wt) -> (B*Ht*Wt,) flat indices
        flat_idx = assignment_map.reshape(-1)
        # (B*Ht*Wt, C, A, A)
        gathered = atoms_clamped[flat_idx]
        # Reshape to (B, Ht, Wt, C, A, A)
        gathered = gathered.reshape(B, Ht, Wt, C, A, A)
        # Permute and reshape to (B, C, Ht*A, Wt*A)
        # (B, Ht, Wt, C, A, A) -> (B, C, Ht, A, Wt, A) -> (B, C, Ht*A, Wt*A)
        gathered = gathered.permute(0, 3, 1, 4, 2, 5).contiguous()
        frame = gathered.reshape(B, C, Ht * A, Wt * A)

        return frame

    def compute_assignment(
        self,
        target_frame: torch.Tensor,
    ) -> torch.Tensor:
        """Find the best atom for each tile position via L2 distance.

        Splits the target frame into non-overlapping patches and assigns
        each patch to its nearest codebook atom.

        Args:
            target_frame: (B, 3, H, W) float tensor in [0, 255].

        Returns:
            (B, H_tiles, W_tiles) long tensor of atom indices.
        """
        B, C, H, W = target_frame.shape
        A = self.atom_size
        Ht, Wt = H // A, W // A

        # Extract non-overlapping patches: (B, C, Ht, A, Wt, A)
        patches = target_frame[:, :, :Ht * A, :Wt * A]
        patches = patches.reshape(B, C, Ht, A, Wt, A)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.reshape(B * Ht * Wt, C * A * A)  # (B*Ht*Wt, D)

        # Atoms flattened: (K, D)
        atoms_flat = self.atoms.detach().clamp(0.0, 255.0).reshape(self.num_atoms, -1)

        # L2 distances: (B*Ht*Wt, K)
        dists = torch.cdist(patches.float(), atoms_flat.float(), p=2)
        indices = dists.argmin(dim=1)  # (B*Ht*Wt,)

        return indices.reshape(B, Ht, Wt)

    def serialize(self) -> bytes:
        """Serialize the codebook to compact bytes (4-bit quantization).

        Returns:
            bytes object containing the quantized codebook.
        """
        atoms = self.atoms.detach().clamp(0.0, 255.0)
        # Quantize to 4 bits: map [0, 255] -> [0, 15]
        quantized = (atoms / 255.0 * 15.0).round().clamp(0, 15).byte()
        flat = quantized.reshape(-1)
        # Pack two 4-bit values per byte
        num_values = flat.shape[0]
        packed_len = (num_values + 1) // 2
        packed = bytearray(packed_len)
        for i in range(0, num_values - 1, 2):
            packed[i // 2] = (flat[i].item() << 4) | flat[i + 1].item()
        if num_values % 2 == 1:
            packed[-1] = flat[-1].item() << 4
        # Header: num_atoms(2B) + atom_size(2B) + num_channels(1B)
        header = struct.pack("<HHB", self.num_atoms, self.atom_size, self.num_channels)
        return header + bytes(packed)

    @classmethod
    def deserialize(cls, data: bytes) -> "TextureAtomCodebook":
        """Deserialize a codebook from packed bytes.

        Args:
            data: bytes object from serialize().

        Returns:
            Reconstructed TextureAtomCodebook instance.
        """
        num_atoms, atom_size, num_channels = struct.unpack("<HHB", data[:5])
        packed = data[5:]
        total_values = num_atoms * num_channels * atom_size * atom_size
        flat = []
        for i in range(len(packed)):
            flat.append((packed[i] >> 4) & 0x0F)
            flat.append(packed[i] & 0x0F)
        flat = flat[:total_values]
        # De-quantize from 4 bits back to [0, 255]
        atoms_tensor = torch.tensor(flat, dtype=torch.float32).reshape(
            num_atoms, num_channels, atom_size, atom_size
        ) * (255.0 / 15.0)
        instance = cls(num_atoms=num_atoms, atom_size=atom_size, num_channels=num_channels)
        instance.atoms = nn.Parameter(atoms_tensor)
        return instance

    def byte_size(self) -> int:
        """Size of the serialized codebook in bytes."""
        return len(self.serialize())


class MotionFieldCodec(nn.Module):
    """Compact representation of how textures flow between frames.

    Instead of storing per-pixel optical flow (expensive), stores a
    low-rank affine motion model per semantic region.  Each of the 5
    semantic classes gets a 6-parameter affine transform (translation,
    rotation, scale, shear).  5 * 6 * 4 bytes = 120 bytes per frame pair.

    For T frames: (T-1) * 120 bytes = ~2.9KB for 25 frames.

    The affine model is sufficient because within each semantic class,
    motion is approximately rigid (the road translates, the car moves
    as a unit, the sky is static).

    Args:
        num_classes: number of semantic classes (5 for comma).
        num_affine_params: parameters per class (6 for 2D affine).
    """

    def __init__(
        self,
        num_classes: int = 5,
        num_affine_params: int = 6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_affine_params = num_affine_params

    def estimate_motion(
        self,
        frame_prev: torch.Tensor,
        frame_curr: torch.Tensor,
        mask_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate per-class affine motion from frame pair.

        Uses least-squares fitting of affine model to optical flow
        within each class region.  The "flow" is approximated by the
        spatial gradient of the intensity difference.

        Args:
            frame_prev: (B, 3, H, W) float previous frame.
            frame_curr: (B, 3, H, W) float current frame.
            mask_prev: (B, H, W) long segmentation mask.

        Returns:
            (B, num_classes, 6) affine parameters per class.
            Each row is [a11, a12, tx, a21, a22, ty] for the 2x3 matrix.
        """
        B, C, H, W = frame_prev.shape
        device = frame_prev.device

        # Intensity difference as flow proxy (grayscale)
        gray_prev = frame_prev.mean(dim=1)  # (B, H, W)
        gray_curr = frame_curr.mean(dim=1)
        diff = gray_curr - gray_prev  # (B, H, W)

        # Spatial gradients of previous frame
        gx = gray_prev[:, :, 1:] - gray_prev[:, :, :-1]  # (B, H, W-1)
        gy = gray_prev[:, 1:, :] - gray_prev[:, :-1, :]  # (B, H-1, W)

        # Pad to original size
        gx = F.pad(gx, (0, 1), mode="replicate")  # (B, H, W)
        gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")  # (B, H, W)

        # Coordinate grid
        ys = torch.arange(H, device=device, dtype=torch.float32) / H
        xs = torch.arange(W, device=device, dtype=torch.float32) / W
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

        params = torch.zeros(B, self.num_classes, 6, device=device)

        for b in range(B):
            for c in range(self.num_classes):
                region = (mask_prev[b] == c)  # (H, W)
                n_pixels = region.sum().item()
                if n_pixels < 10:
                    # Too few pixels — identity transform
                    params[b, c, 0] = 1.0  # a11
                    params[b, c, 4] = 1.0  # a22
                    continue

                # Build linear system: [gx*x, gx*y, gx, gy*x, gy*y, gy] @ p = -diff
                gx_r = gx[b][region]
                gy_r = gy[b][region]
                x_r = grid_x[region]
                y_r = grid_y[region]
                d_r = diff[b][region]

                A_mat = torch.stack([
                    gx_r * x_r, gx_r * y_r, gx_r,
                    gy_r * x_r, gy_r * y_r, gy_r,
                ], dim=1)  # (N, 6)

                # Least squares solve with regularization
                ATA = A_mat.T @ A_mat + 1e-4 * torch.eye(6, device=device)
                ATb = A_mat.T @ (-d_r)
                sol = torch.linalg.solve(ATA, ATb)
                params[b, c] = sol

        return params

    def apply_motion(
        self,
        frame: torch.Tensor,
        mask: torch.Tensor,
        affine_params: torch.Tensor,
    ) -> torch.Tensor:
        """Apply per-class affine motion to warp a frame.

        Args:
            frame: (B, 3, H, W) float frame to warp.
            mask: (B, H, W) long segmentation mask.
            affine_params: (B, num_classes, 6) affine parameters.

        Returns:
            (B, 3, H, W) warped frame.
        """
        B, C, H, W = frame.shape
        device = frame.device

        # Build per-pixel affine grid
        ys = torch.arange(H, device=device, dtype=torch.float32) / H
        xs = torch.arange(W, device=device, dtype=torch.float32) / W
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        warped = torch.zeros_like(frame)

        for b in range(B):
            for c in range(self.num_classes):
                region = (mask[b] == c)
                if region.sum() == 0:
                    continue

                a = affine_params[b, c]
                # (x', y') = A @ (x, y, 1)
                new_x = a[0] * grid_x + a[1] * grid_y + a[2]
                new_y = a[3] * grid_x + a[4] * grid_y + a[5]

                # Convert to grid_sample coords: [-1, 1]
                sample_x = new_x * 2.0 - 1.0
                sample_y = new_y * 2.0 - 1.0

                grid_sample = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)
                sampled = F.grid_sample(
                    frame[b:b + 1], grid_sample,
                    mode="bilinear", padding_mode="border", align_corners=False,
                )

                region_expanded = region.unsqueeze(0).unsqueeze(0).expand(1, C, H, W)
                warped[b:b + 1] = warped[b:b + 1] + sampled * region_expanded.float()

        return warped.clamp(0.0, 255.0)

    def serialize(self, affine_params: torch.Tensor) -> bytes:
        """Serialize affine parameters to bytes (FP16).

        Args:
            affine_params: (T-1, num_classes, 6) float tensor.

        Returns:
            bytes object.
        """
        data = affine_params.detach().cpu().half().numpy().tobytes()
        header = struct.pack("<HBB", affine_params.shape[0], self.num_classes, self.num_affine_params)
        return header + data

    def deserialize(self, data: bytes) -> torch.Tensor:
        """Deserialize affine parameters from bytes.

        Args:
            data: bytes from serialize().

        Returns:
            (T-1, num_classes, 6) float tensor.
        """
        import numpy as np

        T_minus_1, num_classes, num_params = struct.unpack("<HBB", data[:4])
        arr = np.frombuffer(data[4:], dtype=np.float16).reshape(T_minus_1, num_classes, num_params)
        return torch.from_numpy(arr.astype(np.float32))


class ScorerCorrectionTargets(nn.Module):
    """Pre-computed per-pixel adjustments to minimize scorer distortion.

    After the codebook renders a coarse frame, these corrections are added
    to fine-tune specific pixels that the scorer is sensitive to.  The
    corrections are sparse: only pixels near class boundaries (where SegNet
    is most sensitive) and high-gradient regions (where PoseNet is most
    sensitive) get non-zero corrections.

    Storage: sparse format (index + value pairs).  Typically ~1-5% of
    pixels need correction, so sparse storage is much cheaper than dense.

    Args:
        max_corrections_per_frame: maximum number of pixel corrections
            stored per frame.
        value_bits: quantization bits for correction values.
    """

    def __init__(
        self,
        max_corrections_per_frame: int = 2000,
        value_bits: int = 8,
    ):
        super().__init__()
        self.max_corrections_per_frame = max_corrections_per_frame
        self.value_bits = value_bits

    def compute_corrections(
        self,
        rendered_frame: torch.Tensor,
        target_frame: torch.Tensor,
        fragility_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute sparse corrections for the most scorer-sensitive pixels.

        Args:
            rendered_frame: (B, 3, H, W) float — codebook-rendered frame.
            target_frame: (B, 3, H, W) float — desired ground truth.
            fragility_map: (B, H, W) float in [0, 1] — scorer sensitivity.

        Returns:
            indices: (B, K, 2) long — (y, x) coordinates of corrections.
            values: (B, K, 3) float — RGB correction deltas.
            K = min(max_corrections_per_frame, number of nonzero corrections).
        """
        B, C, H, W = rendered_frame.shape
        K = self.max_corrections_per_frame
        device = rendered_frame.device

        residual = target_frame - rendered_frame  # (B, 3, H, W)
        # Weight residuals by fragility to prioritize scorer-sensitive pixels
        importance = fragility_map.unsqueeze(1) * residual.abs()  # (B, 3, H, W)
        importance_score = importance.sum(dim=1)  # (B, H, W)

        all_indices = []
        all_values = []

        for b in range(B):
            flat_importance = importance_score[b].reshape(-1)
            num_select = min(K, (flat_importance > 0).sum().item())
            if num_select == 0:
                all_indices.append(torch.zeros(K, 2, device=device, dtype=torch.long))
                all_values.append(torch.zeros(K, 3, device=device))
                continue

            topk_vals, topk_flat = torch.topk(flat_importance, num_select)
            ys = topk_flat // W
            xs = topk_flat % W
            idx = torch.stack([ys, xs], dim=1)  # (num_select, 2)
            vals = residual[b, :, ys, xs].T  # (num_select, 3)

            # Quantize values
            max_val = vals.abs().max() + 1e-8
            scale = (2 ** (self.value_bits - 1) - 1) / max_val
            quantized = (vals * scale).round() / scale

            # Pad to K if needed
            if num_select < K:
                pad_idx = torch.zeros(K - num_select, 2, device=device, dtype=torch.long)
                pad_val = torch.zeros(K - num_select, 3, device=device)
                idx = torch.cat([idx, pad_idx], dim=0)
                quantized = torch.cat([quantized, pad_val], dim=0)

            all_indices.append(idx)
            all_values.append(quantized)

        return torch.stack(all_indices), torch.stack(all_values)

    def apply_corrections(
        self,
        frame: torch.Tensor,
        indices: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Apply sparse corrections to a rendered frame.

        Args:
            frame: (B, 3, H, W) float — frame to correct.
            indices: (B, K, 2) long — (y, x) correction positions.
            values: (B, K, 3) float — RGB correction deltas.

        Returns:
            (B, 3, H, W) corrected frame in [0, 255].
        """
        result = frame.clone()
        B = frame.shape[0]

        for b in range(B):
            for k in range(indices.shape[1]):
                y, x = indices[b, k, 0].item(), indices[b, k, 1].item()
                if y == 0 and x == 0 and values[b, k].abs().sum() < 1e-8:
                    continue  # Skip padding entries
                result[b, :, y, x] = result[b, :, y, x] + values[b, k]

        return result.clamp(0.0, 255.0)

    def serialize(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
    ) -> bytes:
        """Serialize sparse corrections to bytes.

        Format: uint16 count, then (uint16 y, uint16 x, int8 r, int8 g, int8 b)
        per correction.  7 bytes per correction.

        Args:
            indices: (K, 2) long tensor.
            values: (K, 3) float tensor.

        Returns:
            bytes object.
        """
        # Filter out padding (zero index + zero value)
        mask = (indices.abs().sum(dim=1) > 0) | (values.abs().sum(dim=1) > 1e-8)
        valid_idx = indices[mask]
        valid_val = values[mask]
        count = valid_idx.shape[0]

        buf = struct.pack("<H", count)
        for k in range(count):
            y, x = valid_idx[k, 0].item(), valid_idx[k, 1].item()
            r = max(-128, min(127, int(valid_val[k, 0].item())))
            g = max(-128, min(127, int(valid_val[k, 1].item())))
            b_val = max(-128, min(127, int(valid_val[k, 2].item())))
            buf += struct.pack("<HHbbb", y, x, r, g, b_val)
        return buf


def build_minimal_archive(
    codebook: TextureAtomCodebook,
    motion_params: torch.Tensor | None,
    motion_codec: MotionFieldCodec | None,
    correction_data: tuple[torch.Tensor, torch.Tensor] | None,
    corrections_codec: ScorerCorrectionTargets | None,
    assignment_maps: torch.Tensor | None = None,
) -> bytes:
    """Assemble the complete archive as a zip file.

    Packs the codebook, motion field, corrections, and optionally the
    assignment maps into a single zip archive.  The target size is ~15KB.

    Args:
        codebook: the trained texture atom codebook.
        motion_params: (T-1, num_classes, 6) affine motion parameters,
            or None if no motion is used.
        motion_codec: MotionFieldCodec instance for serialization,
            or None.
        correction_data: (indices, values) tuple from
            ScorerCorrectionTargets.compute_corrections(), or None.
        corrections_codec: ScorerCorrectionTargets instance for
            serialization, or None.
        assignment_maps: (T, H_tiles, W_tiles) long tensor of atom
            assignments per frame, or None (will be computed at inflate
            time).

    Returns:
        bytes of the zip archive.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        # Codebook
        zf.writestr("codebook.bin", codebook.serialize())

        # Motion field
        if motion_params is not None and motion_codec is not None:
            zf.writestr("motion.bin", motion_codec.serialize(motion_params))

        # Corrections (per-frame)
        if correction_data is not None and corrections_codec is not None:
            indices, values = correction_data
            if indices.ndim == 3:
                # (T, K, 2) and (T, K, 3) — serialize each frame
                for t in range(indices.shape[0]):
                    data = corrections_codec.serialize(indices[t], values[t])
                    zf.writestr(f"corrections_{t:04d}.bin", data)
            else:
                # Single frame
                zf.writestr("corrections_0000.bin",
                            corrections_codec.serialize(indices, values))

        # Assignment maps (compact: 1 byte per tile, since num_atoms <= 256)
        if assignment_maps is not None:
            assign_bytes = assignment_maps.detach().cpu().byte().numpy().tobytes()
            zf.writestr("assignments.bin", assign_bytes)

    return buf.getvalue()


# ── Smoke tests ───────────────────────────────────────────────────────


def _smoke_test() -> None:
    """Run basic shape, serialization, and forward-pass checks."""
    B, H, W = 2, 64, 64
    atom_size = 16
    num_atoms = 32
    num_classes = 5

    # TextureAtomCodebook
    codebook = TextureAtomCodebook(num_atoms=num_atoms, atom_size=atom_size)

    Ht, Wt = H // atom_size, W // atom_size
    assignment = torch.randint(0, num_atoms, (B, Ht, Wt))
    rendered = codebook(assignment)
    assert rendered.shape == (B, 3, H, W), f"Rendered shape: {rendered.shape}"
    assert rendered.min() >= 0.0 and rendered.max() <= 255.0

    # Assignment computation
    target = torch.rand(B, 3, H, W) * 255.0
    computed_assign = codebook.compute_assignment(target)
    assert computed_assign.shape == (B, Ht, Wt)
    assert computed_assign.min() >= 0
    assert computed_assign.max() < num_atoms

    # Serialization round-trip
    serialized = codebook.serialize()
    assert len(serialized) < 50_000, f"Codebook too large: {len(serialized)} bytes"
    restored = TextureAtomCodebook.deserialize(serialized)
    assert restored.num_atoms == num_atoms
    assert restored.atom_size == atom_size
    # Check round-trip fidelity (4-bit quantization loses precision)
    orig_q = (codebook.atoms.detach().clamp(0, 255) / 255.0 * 15.0).round() * (255.0 / 15.0)
    restored_atoms = restored.atoms.detach()
    assert (orig_q - restored_atoms).abs().max() < 20.0, "Round-trip error too large"
    print(f"  archive_codec: codebook verified ({len(serialized)} bytes)")

    # MotionFieldCodec
    motion = MotionFieldCodec(num_classes=num_classes)
    frame_a = torch.rand(B, 3, H, W) * 255.0
    frame_b = torch.rand(B, 3, H, W) * 255.0
    mask_a = torch.randint(0, num_classes, (B, H, W))
    params = motion.estimate_motion(frame_a, frame_b, mask_a)
    assert params.shape == (B, num_classes, 6)

    warped = motion.apply_motion(frame_a, mask_a, params)
    assert warped.shape == (B, 3, H, W)

    motion_bytes = motion.serialize(params)
    assert len(motion_bytes) < 1000
    restored_params = motion.deserialize(motion_bytes)
    assert restored_params.shape == params.shape
    print(f"  archive_codec: motion codec verified ({len(motion_bytes)} bytes)")

    # ScorerCorrectionTargets
    corrections = ScorerCorrectionTargets(max_corrections_per_frame=100)
    fragility = torch.rand(B, H, W)
    indices, values = corrections.compute_corrections(rendered, target, fragility)
    assert indices.shape[0] == B and indices.shape[2] == 2
    assert values.shape[0] == B and values.shape[2] == 3

    corrected = corrections.apply_corrections(rendered, indices, values)
    assert corrected.shape == (B, 3, H, W)
    assert corrected.min() >= 0.0 and corrected.max() <= 255.0

    corr_bytes = corrections.serialize(indices[0], values[0])
    assert len(corr_bytes) < 10_000
    print(f"  archive_codec: corrections verified ({len(corr_bytes)} bytes)")

    # build_minimal_archive
    archive_bytes = build_minimal_archive(
        codebook=codebook,
        motion_params=params,
        motion_codec=motion,
        correction_data=(indices, values),
        corrections_codec=corrections,
        assignment_maps=computed_assign,
    )
    assert len(archive_bytes) < 100_000, f"Archive too large: {len(archive_bytes)} bytes"
    # Verify it's a valid zip
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
        names = zf.namelist()
        assert "codebook.bin" in names
        assert "motion.bin" in names
        assert "assignments.bin" in names
    print(f"  archive_codec: minimal archive verified ({len(archive_bytes)} bytes)")

    print("archive_codec: all smoke tests passed")


if __name__ == "__main__":
    _smoke_test()
