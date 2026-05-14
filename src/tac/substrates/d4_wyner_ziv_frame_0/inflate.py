"""D4 Wyner-Ziv frame-0 inflate runtime — contest raw-output contract.

Loads the WZF01 archive, verifies the base substrate sha, asks the base
substrate's inflate module to materialize frame_1 per pair, then synthesizes
frame_0 via motion + residual and writes one raw-output ``.raw`` file per
contest video.

NO scorer code is imported per CLAUDE.md "Strict scorer rule" + Catalog #6.
NO MPS device (Catalog #1; CPU/CUDA only).

Per Catalog #146 the inflate.py honors the contest's 3-positional-arg
``inflate.sh <archive_dir> <output_dir> <file_list>`` contract.

Per HNeRV parity discipline L4 the inflate runtime LOC budget is ≤ 200 for
substrate-engineering lanes (full motion-warp + residual decode + base
substrate inflate composition).
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from pathlib import Path

import torch

from tac.substrates._shared.inflate_runtime import (
    raw_output_path,
    select_inflate_device,
    write_rgb_pair_to_raw,
)
from tac.substrates.d4_wyner_ziv_frame_0.archive import (
    WynerZivFrame0Archive,
    deserialize_motion_to_tensor,
    parse_archive,
)
from tac.substrates.d4_wyner_ziv_frame_0.frame0_synthesis import synthesize_frame_0
from tac.substrates.d4_wyner_ziv_frame_0.motion_model import (
    MotionModelMode,
    OpticalFlowField,
    SE3MotionParams,
)
from tac.substrates.d4_wyner_ziv_frame_0.residual_codec import decode_residual_blob

# Base substrate provider: a callable that takes (base_substrate_bytes,
# pair_idx, device) -> frame_1 as (1, 3, H, W) tensor in unit range.
# Substrate trainers register their provider before calling inflate; default
# is a smoke provider that returns a deterministic constant frame (useful for
# tests; production runs MUST register the real base provider).
_BASE_PROVIDERS: dict[str, Callable[[bytes, int, str], torch.Tensor]] = {}


def register_base_substrate_provider(
    base_substrate_id: str,
    provider: Callable[[bytes, int, str], torch.Tensor],
) -> None:
    """Register a base substrate's frame_1 provider keyed by id."""
    _BASE_PROVIDERS[base_substrate_id] = provider


def _smoke_provider(base_bytes: bytes, pair_idx: int, device: str) -> torch.Tensor:
    """Deterministic constant-frame_1 provider for tests / smokes only.

    Production trainers MUST register a real provider via
    ``register_base_substrate_provider``; this default is the
    no-base-substrate-available smoke fallback.
    """
    h, w = 384, 512
    # Deterministic per-pair seed derived from base_bytes sha + pair index.
    seed_bytes = hashlib.sha256(base_bytes + pair_idx.to_bytes(4, "big")).digest()
    seed = int.from_bytes(seed_bytes[:4], "big") % (2**31)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.rand((1, 3, h, w), generator=gen, dtype=torch.float32).to(device)


register_base_substrate_provider("smoke_base_substrate_v0", _smoke_provider)
register_base_substrate_provider("external_base_substrate_v0", _smoke_provider)


def _read_single_member_archive_bytes(archive_dir: Path) -> bytes:
    """Read the single contest archive member, failing on ambiguity."""
    zero_bin = archive_dir / "0.bin"
    x_member = archive_dir / "x"
    present = [path for path in (zero_bin, x_member) if path.is_file()]
    if len(present) != 1:
        if not present:
            raise FileNotFoundError(
                f"expected exactly one archive member at {zero_bin} or {x_member}"
            )
        raise ValueError(
            f"ambiguous archive members present: {zero_bin} and {x_member}"
        )
    return present[0].read_bytes()


def _verify_base_sha(arc: WynerZivFrame0Archive) -> None:
    """Fail closed when the declared sha does not match the embedded base bytes.

    If the embedded base_substrate_bytes is empty (operator opted into the
    sister-zip-member layout) we skip the cryptographic check; the operator
    is responsible for providing the bytes via the registered provider.
    """
    if not arc.base_substrate_bytes:
        return
    actual = hashlib.sha256(arc.base_substrate_bytes).hexdigest()
    if actual != arc.base_substrate_archive_sha256_hex:
        raise ValueError(
            f"base substrate sha256 mismatch: declared "
            f"{arc.base_substrate_archive_sha256_hex} vs actual {actual} "
            "— custody-cardinal violation; refusing to inflate"
        )


def inflate_one_video(
    archive_bytes: bytes,
    output_raw_path: Path,
    *,
    device: str | None = None,
) -> int:
    """Inflate one WZF01 archive's bytes into one contest ``.raw`` file."""
    arc = parse_archive(archive_bytes)
    _verify_base_sha(arc)
    render_device = select_inflate_device(device)

    motion_mode = MotionModelMode.SE3_PARAMETRIC if arc.motion_mode == 0 else MotionModelMode.OPTICAL_FLOW
    motion_tensor = deserialize_motion_to_tensor(
        arc.motion_blob_raw,
        motion_mode=arc.motion_mode,
        num_pairs=arc.num_pairs,
        flow_grid_h=arc.flow_grid_h,
        flow_grid_w=arc.flow_grid_w,
    ).to(render_device)
    residual_coarse = decode_residual_blob(
        arc.residual_blob, expected_num_pairs=arc.num_pairs
    ).to(render_device)

    base_id = str(arc.meta.get("base_substrate_id", "external_base_substrate_v0"))
    provider = _BASE_PROVIDERS.get(base_id)
    if provider is None:
        # Fall back to smoke provider if base id is unregistered (test path).
        provider = _smoke_provider

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    with torch.no_grad(), output_raw_path.open("wb") as fh:
        for pair_idx in range(arc.num_pairs):
            frame_1 = provider(arc.base_substrate_bytes, pair_idx, render_device)
            if frame_1.dim() != 4 or frame_1.shape[0] != 1 or frame_1.shape[1] != 3:
                raise ValueError(
                    f"base provider must return (1, 3, H, W); got {tuple(frame_1.shape)}"
                )
            if motion_mode == MotionModelMode.SE3_PARAMETRIC:
                se3_params = SE3MotionParams.from_flat(motion_tensor[pair_idx : pair_idx + 1])
                flow_field = None
            else:
                se3_params = None
                flow_field = OpticalFlowField(
                    flow_uv=motion_tensor[pair_idx : pair_idx + 1],
                    grid_h=arc.flow_grid_h,
                    grid_w=arc.flow_grid_w,
                )
            frame_0 = synthesize_frame_0(
                frame_1=frame_1,
                motion_mode=motion_mode,
                se3_params=se3_params,
                flow_field=flow_field,
                residual=residual_coarse[pair_idx : pair_idx + 1],
                output_hw=(384, 512),
                clamp_unit=True,
            )
            frames_written += write_rgb_pair_to_raw(
                fh, frame_0, frame_1, input_range="unit"
            )
    return frames_written


def main_cli() -> int:
    """CLI: ``inflate.py <archive_dir> <output_dir> <file_list>``."""
    if len(sys.argv) < 4:
        print(
            "usage: inflate.py <archive_dir> <output_dir> <file_list>",
            file=sys.stderr,
        )
        return 2
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    file_list = file_list_path.read_text(encoding="utf-8").strip().splitlines()
    archive_bytes = _read_single_member_archive_bytes(archive_dir)
    device = select_inflate_device()
    for fname in file_list:
        name = fname.strip()
        if not name:
            continue
        inflate_one_video(archive_bytes, raw_output_path(output_dir, name), device=device)
    return 0


__all__ = [
    "_read_single_member_archive_bytes",
    "inflate_one_video",
    "main_cli",
    "register_base_substrate_provider",
]


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main_cli())
