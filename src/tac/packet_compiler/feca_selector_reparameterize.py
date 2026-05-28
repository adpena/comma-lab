# SPDX-License-Identifier: MIT
"""Build exact-roundtrip FECa selector reparameterization candidates.

The current PR110-family packet stores a FECa/FEC10 arithmetic-coded selector
inside the FP11 member.  This module only changes the selector codec parameters
and the matching runtime constant, then proves the decoded selector codes,
source payload, and DQS1 tail are unchanged.  It is a rate-only packet
transform; it does not claim score authority.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import struct
import sys
import time
import zipfile
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

from tac.optimization.entropy_position import classify_entropy_position
from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    require_no_truthy_authority_fields,
)
from tac.repo_io import sha256_bytes, sha256_file, tree_sha256, write_json_artifact

FECA_REPARAMETERIZATION_MANIFEST_SCHEMA = "feca_selector_reparameterization_manifest.v1"
FECA_REPARAMETERIZATION_PROOF_SCHEMA = "feca_selector_runtime_consumption_proof.v1"
FECA_SELECTOR_TARGET_KIND = "selector_stream_context_recode_v1"
FECA_SELECTOR_MATERIALIZER_ID = "feca_selector_reparameterize_adapter"
FECA_SELECTOR_RECEIVER_CONTRACT_ID = "selector_stream_context_recode_v1.receiver.v1"
FECA_SELECTOR_RECEIVER_CONTRACT_KIND = "source_runtime_native_selector_context_recode"
FP11_MAGIC = b"FP11"
FECA_MAGIC = b"FECa"
FEC8_MAGIC = b"FEC8"
SELECTOR_CODEC_FEC10_ADAPTIVE_BLEND = "fec10_adaptive_blend"
SELECTOR_CODEC_FEC8_STATIC_ORDER1 = "fec8_markov_static_order1"
SELECTOR_CODEC_FEC8_ADAPTIVE_ORDER1 = "fec8_markov_adaptive_order1"
SELECTOR_CODEC_FEC8_STATIC_ORDER2 = "fec8_markov_static_order2"
DEFAULT_SELECTOR_CODEC_FAMILIES = (
    SELECTOR_CODEC_FEC10_ADAPTIVE_BLEND,
    SELECTOR_CODEC_FEC8_STATIC_ORDER1,
    SELECTOR_CODEC_FEC8_ADAPTIVE_ORDER1,
    SELECTOR_CODEC_FEC8_STATIC_ORDER2,
)
_FEC8_CODEC_FAMILIES = {
    SELECTOR_CODEC_FEC8_STATIC_ORDER1,
    SELECTOR_CODEC_FEC8_ADAPTIVE_ORDER1,
    SELECTOR_CODEC_FEC8_STATIC_ORDER2,
}


class FecaSelectorReparameterizationError(ValueError):
    """Raised when a FECa selector reparameterization candidate is unsafe."""


@contextmanager
def _prepended_sys_path(path: Path):
    value = str(path)
    inserted = False
    if value not in sys.path:
        sys.path.insert(0, value)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(value)
            except ValueError:
                pass


def _load_feca_module(encoder_dir: Path, *, module_suffix: str) -> ModuleType:
    module_path = encoder_dir / "build_pr101_frame_exploit_selector_packet_fec10_hybrid.py"
    if not module_path.is_file():
        raise FecaSelectorReparameterizationError(f"missing FECa encoder module: {module_path}")
    spec = importlib.util.spec_from_file_location(
        f"feca_selector_reparameterize_{module_suffix}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise FecaSelectorReparameterizationError(f"could not load FECa module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    original_dont_write_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        with _prepended_sys_path(encoder_dir):
            spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = original_dont_write_bytecode
    return module


def _load_markov_module(encoder_dir: Path, *, module_suffix: str) -> ModuleType:
    module_path = encoder_dir / "build_pr101_frame_exploit_selector_packet_markov.py"
    if not module_path.is_file():
        raise FecaSelectorReparameterizationError(
            f"missing FEC8 Markov encoder module: {module_path}"
        )
    spec = importlib.util.spec_from_file_location(
        f"feca_selector_markov_{module_suffix}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise FecaSelectorReparameterizationError(
            f"could not load FEC8 Markov module: {module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    original_dont_write_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        with _prepended_sys_path(encoder_dir):
            spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = original_dont_write_bytecode
    return module


def _read_single_member(archive_path: Path) -> tuple[zipfile.ZipInfo, bytes]:
    with zipfile.ZipFile(archive_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise FecaSelectorReparameterizationError(
                f"expected one archive member, found {len(infos)}"
            )
        info = infos[0]
        return info, archive.read(info.filename)


def split_fp11_member(
    member_payload: bytes,
    *,
    allowed_selector_magics: tuple[bytes, ...] = (FECA_MAGIC,),
) -> dict[str, Any]:
    if len(member_payload) < 10 or member_payload[:4] != FP11_MAGIC:
        raise FecaSelectorReparameterizationError("archive member is not an FP11 packet")
    source_len = struct.unpack_from("<I", member_payload, 4)[0]
    selector_len_offset = 8 + source_len
    if selector_len_offset + 2 > len(member_payload):
        raise FecaSelectorReparameterizationError("FP11 packet truncated before selector length")
    selector_len = struct.unpack_from("<H", member_payload, selector_len_offset)[0]
    selector_start = selector_len_offset + 2
    selector_end = selector_start + selector_len
    if selector_end > len(member_payload):
        raise FecaSelectorReparameterizationError("FP11 selector payload is truncated")
    selector_payload = member_payload[selector_start:selector_end]
    if not selector_payload.startswith(allowed_selector_magics):
        raise FecaSelectorReparameterizationError(
            "expected selector payload magic "
            f"{tuple(magic.decode('ascii', errors='replace') for magic in allowed_selector_magics)}, "
            f"got {selector_payload[:4]!r}"
        )
    return {
        "source_payload": member_payload[8:selector_len_offset],
        "selector_payload": selector_payload,
        "dqs1_tail": member_payload[selector_end:],
        "source_len": source_len,
        "selector_len": selector_len,
    }


def join_fp11_member(*, source_payload: bytes, selector_payload: bytes, dqs1_tail: bytes) -> bytes:
    if len(source_payload) > 0xFFFFFFFF:
        raise FecaSelectorReparameterizationError("source payload too large for FP11")
    if len(selector_payload) > 0xFFFF:
        raise FecaSelectorReparameterizationError("selector payload too large for FP11")
    return (
        FP11_MAGIC
        + struct.pack("<I", len(source_payload))
        + source_payload
        + struct.pack("<H", len(selector_payload))
        + selector_payload
        + dqs1_tail
    )


def _encode_with_params(
    module: ModuleType,
    codes: Sequence[int],
    *,
    scale: int,
    alpha: int,
) -> bytes:
    original = (
        module.ALPHA_DEFAULT,
        module._BlendContextModel.SCALE,
        module._PRIOR_MODEL,
        module._CTX1_MODELS,
        module._CTX_BLEND_MODELS,
        module._CTX2_ROW_SUMS,
    )
    try:
        module.ALPHA_DEFAULT = int(alpha)
        module._BlendContextModel.SCALE = int(scale)
        (
            module._PRIOR_MODEL,
            module._CTX1_MODELS,
            module._CTX_BLEND_MODELS,
            module._CTX2_ROW_SUMS,
        ) = module._build_priors()
        payload = module.encode_fec10_hybrid_adaptive_blend(codes, n_pairs=len(codes))
        decoded = module.decode_fec10_hybrid_selector(payload)
        if list(decoded) != list(codes):
            raise FecaSelectorReparameterizationError(
                f"FECa scale={scale} alpha={alpha} failed selector-code roundtrip"
            )
        return payload
    finally:
        (
            module.ALPHA_DEFAULT,
            module._BlendContextModel.SCALE,
            module._PRIOR_MODEL,
            module._CTX1_MODELS,
            module._CTX_BLEND_MODELS,
            module._CTX2_ROW_SUMS,
        ) = original


def _encode_fec8_markov(module: ModuleType, codes: Sequence[int], *, codec_family: str) -> bytes:
    if codec_family == SELECTOR_CODEC_FEC8_STATIC_ORDER1:
        payload = module.encode_fec8_markov_selector_static(codes, n_pairs=len(codes))
    elif codec_family == SELECTOR_CODEC_FEC8_ADAPTIVE_ORDER1:
        payload = module.encode_fec8_markov_selector_adaptive(codes, n_pairs=len(codes))
    elif codec_family == SELECTOR_CODEC_FEC8_STATIC_ORDER2:
        payload = module.encode_fec8_markov_selector_static_second_order(
            codes,
            n_pairs=len(codes),
        )
    else:
        raise FecaSelectorReparameterizationError(
            f"unsupported FEC8 selector codec family: {codec_family}"
        )
    decoded = module.decode_fec8_markov_selector(payload)
    if list(decoded) != list(codes):
        raise FecaSelectorReparameterizationError(
            f"{codec_family} failed selector-code roundtrip"
        )
    return payload


def _decode_selector_payload(
    *,
    feca_module: ModuleType,
    markov_module: ModuleType | None,
    payload: bytes,
) -> list[int]:
    if payload.startswith(FECA_MAGIC):
        return list(feca_module.decode_fec10_hybrid_selector(payload))
    if payload.startswith(FEC8_MAGIC):
        if markov_module is None:
            raise FecaSelectorReparameterizationError(
                "FEC8 selector payload requires Markov decoder module"
            )
        return list(markov_module.decode_fec8_markov_selector(payload))
    raise FecaSelectorReparameterizationError(
        f"unsupported selector payload magic: {payload[:4]!r}"
    )


def _selector_codec_order(codec_family: str) -> int:
    if codec_family == SELECTOR_CODEC_FEC10_ADAPTIVE_BLEND:
        return 2
    if codec_family in {
        SELECTOR_CODEC_FEC8_STATIC_ORDER1,
        SELECTOR_CODEC_FEC8_ADAPTIVE_ORDER1,
    }:
        return 1
    if codec_family == SELECTOR_CODEC_FEC8_STATIC_ORDER2:
        return 2
    return 0


def _selector_codec_model_kind(codec_family: str) -> str:
    if codec_family == SELECTOR_CODEC_FEC10_ADAPTIVE_BLEND:
        return "adaptive_blend_prior_ctx1_ctx2"
    if codec_family == SELECTOR_CODEC_FEC8_STATIC_ORDER1:
        return "static_empirical_first_order_markov"
    if codec_family == SELECTOR_CODEC_FEC8_ADAPTIVE_ORDER1:
        return "adaptive_online_first_order_markov"
    if codec_family == SELECTOR_CODEC_FEC8_STATIC_ORDER2:
        return "static_empirical_second_order_markov"
    return "unknown"


def _selector_codec_jobs(
    codec_families: Sequence[str],
    scales: Sequence[int],
    alphas: Sequence[int],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    for codec_family in codec_families:
        codec = str(codec_family).strip()
        if not codec:
            continue
        if codec == SELECTOR_CODEC_FEC10_ADAPTIVE_BLEND:
            for scale in scales:
                for alpha in alphas:
                    key = (codec, int(scale), int(alpha))
                    if key in seen:
                        continue
                    seen.add(key)
                    jobs.append(
                        {
                            "codec_family": codec,
                            "scale": int(scale),
                            "alpha": int(alpha),
                        }
                    )
        elif codec in _FEC8_CODEC_FAMILIES:
            key = (codec, None, None)
            if key in seen:
                continue
            seen.add(key)
            jobs.append({"codec_family": codec, "scale": None, "alpha": None})
        else:
            raise FecaSelectorReparameterizationError(
                f"unknown selector codec family: {codec}"
            )
    if not jobs:
        raise FecaSelectorReparameterizationError("no selector codec families requested")
    return jobs


def _patch_runtime_module(path: Path, *, scale: int, alpha: int) -> None:
    text = path.read_text(encoding="utf-8")
    if "ALPHA_DEFAULT = 2" not in text:
        raise FecaSelectorReparameterizationError("FECa runtime alpha constant not found")
    if "SCALE = 1 << 14" not in text:
        raise FecaSelectorReparameterizationError("FECa runtime scale constant not found")
    text = text.replace("ALPHA_DEFAULT = 2", f"ALPHA_DEFAULT = {int(alpha)}", 1)
    text = text.replace("SCALE = 1 << 14", f"SCALE = {int(scale)}", 1)
    path.write_text(text, encoding="utf-8")


def _write_fec8_decoder_shim(path: Path) -> None:
    path.write_text(
        '''"""Inflate-side decoder shim for FEC8 Markov selector streams."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ENCODER_DIR = _HERE.parent / "encoder"
if str(_ENCODER_DIR) not in sys.path:
    sys.path.insert(0, str(_ENCODER_DIR))

from build_pr101_frame_exploit_selector_packet_markov import (  # type: ignore[import-not-found]  # noqa: E402
    FEC8_MAGIC,
    FEC8_VARIANT_ADAPTIVE,
    FEC8_VARIANT_STATIC,
    FEC8_VARIANT_STATIC_SECOND_ORDER,
    PALETTE_K,
    decode_fec8_markov_selector,
)

__all__ = [
    "FEC8_MAGIC",
    "FEC8_VARIANT_ADAPTIVE",
    "FEC8_VARIANT_STATIC",
    "FEC8_VARIANT_STATIC_SECOND_ORDER",
    "PALETTE_K",
    "decode_fec8_markov_selector",
]
''',
        encoding="utf-8",
    )


def _patch_runtime_fec8_dispatch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if 'selector_payload[:4] == b"FEC8"' not in text:
        needle = (
            '    if selector_payload[:4] == b"FECa":\n'
            '        # V14-V2: Cascade A FEC10 hybrid adaptive-blend dispatch (per RECOVERY-1 commit 39c76755b)\n'
            '        from fec10_hybrid_decoder import decode_fec10_hybrid_selector  # noqa: PLC0415\n'
            '        codes = decode_fec10_hybrid_selector(selector_payload)\n'
            '        specs = tuple(mode_spec_from_static_mode_id(mode_id) for mode_id in FEC6_FIXED_K16_MODE_IDS)\n'
            '        return codes, specs\n'
        )
        replacement = needle + (
            '    if selector_payload[:4] == b"FEC8":\n'
            '        from fec8_markov_decoder import decode_fec8_markov_selector  # noqa: PLC0415\n'
            '        codes = decode_fec8_markov_selector(selector_payload)\n'
            '        specs = tuple(mode_spec_from_static_mode_id(mode_id) for mode_id in FEC6_FIXED_K16_MODE_IDS)\n'
            '        return codes, specs\n'
        )
        if needle not in text:
            raise FecaSelectorReparameterizationError(
                "FECa runtime dispatch block not found for FEC8 patch"
            )
        text = text.replace(needle, replacement, 1)
    text = text.replace(
        'selector_payload.startswith((b"FEC2", b"FEC3", b"FEC5", b"FEC6", b"FECa"))',
        'selector_payload.startswith((b"FEC2", b"FEC3", b"FEC5", b"FEC6", b"FECa", b"FEC8"))',
        1,
    )
    path.write_text(text, encoding="utf-8")


def _patch_inflate_no_bytecode(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "PYTHONDONTWRITEBYTECODE" in text:
        return
    lines = text.splitlines(keepends=True)
    export_line = "export PYTHONDONTWRITEBYTECODE=1\n"
    if lines and lines[0].startswith("#!"):
        lines.insert(1, export_line)
        patched = "".join(lines)
    else:
        patched = export_line + text
    path.write_text(patched, encoding="utf-8")


def _write_stored_archive(archive_path: Path, *, member_name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(member_name)
    info.compress_type = zipfile.ZIP_STORED
    info.date_time = (1980, 1, 1, 0, 0, 0)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(info, payload)


def _archive_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_full_frame_parity_proof(path: str | Path | None) -> tuple[dict[str, Any] | None, Path | None]:
    if path is None:
        return None, None
    proof_path = Path(path)
    if not proof_path.is_file():
        raise FecaSelectorReparameterizationError(
            f"full-frame inflate parity proof missing: {proof_path}"
        )
    payload = json.loads(proof_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FecaSelectorReparameterizationError(
            f"full-frame inflate parity proof must be a JSON object: {proof_path}"
        )
    if payload.get("full_frame_inflate_output_parity_claim") is not True:
        raise FecaSelectorReparameterizationError(
            "full-frame inflate parity proof does not claim full-frame parity"
        )
    if payload.get("cmp_equal") is not True or payload.get("output_sha256_match") is not True:
        raise FecaSelectorReparameterizationError(
            "full-frame inflate parity proof is not byte-identical"
        )
    blockers = payload.get("blockers")
    if isinstance(blockers, list) and blockers:
        raise FecaSelectorReparameterizationError(
            "full-frame inflate parity proof carries blockers: "
            + ",".join(str(item) for item in blockers)
        )
    return payload, proof_path


def build_feca_selector_reparameterized_candidate(
    *,
    source_submission_dir: str | Path,
    output_dir: str | Path,
    codec_families: Sequence[str] = DEFAULT_SELECTOR_CODEC_FAMILIES,
    scales: Sequence[int] = (256, 512, 1024, 2048, 4096, 8192, 16384),
    alphas: Sequence[int] = tuple(range(1, 17)),
    upstream_entropy_positions: Sequence[str] = ("P19", "P18"),
    downstream_materializer_targets: Sequence[str] = ("archive_zip_repack_v1",),
    chain_parent_artifact: str | Path | None = None,
    chain_label: str = "cascade_c_p19_p18_to_p11_selector_context_recode",
    full_frame_inflate_parity_proof: str | Path | None = None,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    source_dir = Path(source_submission_dir)
    output = Path(output_dir)
    if not source_dir.is_dir():
        raise FecaSelectorReparameterizationError(f"missing source submission dir: {source_dir}")
    parity_payload, parity_source_path = _load_full_frame_parity_proof(
        full_frame_inflate_parity_proof
    )
    if output.exists():
        if not allow_overwrite:
            raise FecaSelectorReparameterizationError(f"output exists: {output}")
        shutil.rmtree(output)
    candidate_dir = output / "submission_dir"
    shutil.copytree(
        source_dir,
        candidate_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    _patch_inflate_no_bytecode(candidate_dir / "inflate.sh")

    source_archive = source_dir / "archive.zip"
    candidate_archive = candidate_dir / "archive.zip"
    info, member_payload = _read_single_member(source_archive)
    parts = split_fp11_member(member_payload)
    source_module = _load_feca_module(source_dir / "encoder", module_suffix="source")
    markov_module = (
        _load_markov_module(source_dir / "encoder", module_suffix="source")
        if any(str(family).strip() in _FEC8_CODEC_FAMILIES for family in codec_families)
        else None
    )
    source_codes = _decode_selector_payload(
        feca_module=source_module,
        markov_module=markov_module,
        payload=parts["selector_payload"],
    )

    rows: list[dict[str, Any]] = []
    best_payload = parts["selector_payload"]
    best_scale = int(source_module._BlendContextModel.SCALE)
    best_alpha = int(source_module.ALPHA_DEFAULT)
    best_codec_family = SELECTOR_CODEC_FEC10_ADAPTIVE_BLEND
    jobs = _selector_codec_jobs(codec_families, scales, alphas)
    for job in jobs:
        codec_family = str(job["codec_family"])
        scale = job.get("scale")
        alpha = job.get("alpha")
        try:
            if codec_family == SELECTOR_CODEC_FEC10_ADAPTIVE_BLEND:
                assert scale is not None
                assert alpha is not None
                payload = _encode_with_params(
                    source_module,
                    source_codes,
                    scale=int(scale),
                    alpha=int(alpha),
                )
            else:
                if markov_module is None:
                    markov_module = _load_markov_module(
                        source_dir / "encoder",
                        module_suffix="source",
                    )
                payload = _encode_fec8_markov(
                    markov_module,
                    source_codes,
                    codec_family=codec_family,
                )
        except FecaSelectorReparameterizationError as exc:
            rows.append(
                {
                    "codec_family": codec_family,
                    "context_order": _selector_codec_order(codec_family),
                    "context_model_kind": _selector_codec_model_kind(codec_family),
                    "scale": int(scale) if scale is not None else None,
                    "alpha": int(alpha) if alpha is not None else None,
                    "status": "roundtrip_failed",
                    "error": str(exc),
                }
            )
            continue
        row = {
            "codec_family": codec_family,
            "context_order": _selector_codec_order(codec_family),
            "context_model_kind": _selector_codec_model_kind(codec_family),
            "scale": int(scale) if scale is not None else None,
            "alpha": int(alpha) if alpha is not None else None,
            "status": "roundtrip_equal",
            "selector_payload_bytes": len(payload),
            "selector_payload_sha256": sha256_bytes(payload),
            "selector_saved_bytes": len(parts["selector_payload"]) - len(payload),
            "receiver_support": (
                "native_feca_runtime_constant_patch"
                if codec_family == SELECTOR_CODEC_FEC10_ADAPTIVE_BLEND
                else "candidate_runtime_fec8_dispatch_patch"
            ),
            "byte_closed_candidate_emittable": True,
        }
        rows.append(row)
        best_key = (
            len(best_payload),
            best_codec_family,
            best_scale,
            best_alpha,
        )
        candidate_key = (
            len(payload),
            codec_family,
            int(scale) if scale is not None else -1,
            int(alpha) if alpha is not None else -1,
        )
        if candidate_key < best_key:
            best_payload = payload
            best_codec_family = codec_family
            best_scale = int(scale) if scale is not None else best_scale
            best_alpha = int(alpha) if alpha is not None else best_alpha

    if len(best_payload) >= len(parts["selector_payload"]):
        raise FecaSelectorReparameterizationError(
            "no rate-positive selector context recode found"
        )

    if best_codec_family == SELECTOR_CODEC_FEC10_ADAPTIVE_BLEND:
        _patch_runtime_module(
            candidate_dir / "encoder" / "build_pr101_frame_exploit_selector_packet_fec10_hybrid.py",
            scale=best_scale,
            alpha=best_alpha,
        )
        candidate_runtime_patch = {
            "codec_family": best_codec_family,
            "alpha_default": best_alpha,
            "blend_context_model_scale": best_scale,
            "fec8_dispatch_added": False,
        }
    else:
        _write_fec8_decoder_shim(candidate_dir / "src" / "fec8_markov_decoder.py")
        _patch_runtime_fec8_dispatch(candidate_dir / "inflate.py")
        candidate_runtime_patch = {
            "codec_family": best_codec_family,
            "alpha_default": None,
            "blend_context_model_scale": None,
            "fec8_dispatch_added": True,
        }
    candidate_member = join_fp11_member(
        source_payload=parts["source_payload"],
        selector_payload=best_payload,
        dqs1_tail=parts["dqs1_tail"],
    )
    _write_stored_archive(candidate_archive, member_name=info.filename, payload=candidate_member)

    candidate_module = _load_feca_module(candidate_dir / "encoder", module_suffix="candidate")
    candidate_markov_module = (
        _load_markov_module(candidate_dir / "encoder", module_suffix="candidate")
        if best_codec_family in _FEC8_CODEC_FAMILIES
        else None
    )
    candidate_parts = split_fp11_member(
        _read_single_member(candidate_archive)[1],
        allowed_selector_magics=(FECA_MAGIC, FEC8_MAGIC),
    )
    candidate_codes = _decode_selector_payload(
        feca_module=candidate_module,
        markov_module=candidate_markov_module,
        payload=candidate_parts["selector_payload"],
    )
    if list(candidate_codes) != list(source_codes):
        raise FecaSelectorReparameterizationError("candidate runtime selector decode changed codes")
    if candidate_parts["source_payload"] != parts["source_payload"]:
        raise FecaSelectorReparameterizationError("candidate changed source payload bytes")
    if candidate_parts["dqs1_tail"] != parts["dqs1_tail"]:
        raise FecaSelectorReparameterizationError("candidate changed DQS1 tail bytes")

    archive_saved_bytes = source_archive.stat().st_size - candidate_archive.stat().st_size
    source_archive_record = _archive_record(source_archive)
    candidate_archive_record = _archive_record(candidate_archive)
    if parity_payload is not None:
        right = parity_payload.get("right") if isinstance(parity_payload.get("right"), dict) else {}
        left = parity_payload.get("left") if isinstance(parity_payload.get("left"), dict) else {}
        if right.get("archive_sha256") != candidate_archive_record["sha256"]:
            raise FecaSelectorReparameterizationError(
                "full-frame parity proof right archive SHA does not match candidate"
            )
        if left.get("archive_sha256") != source_archive_record["sha256"]:
            raise FecaSelectorReparameterizationError(
                "full-frame parity proof left archive SHA does not match source"
            )
    candidate_runtime_tree_sha = tree_sha256(candidate_dir)
    proof_path = output / "feca_selector_runtime_consumption_proof.json"
    parity_copy_path = output / "feca_selector_full_frame_inflate_parity_proof.json"
    parity_copy_record: dict[str, Any] | None = None
    if parity_payload is not None:
        write_json_artifact(parity_copy_path, parity_payload)
        parity_copy_record = _archive_record(parity_copy_path)
    inflate_parity_satisfied = parity_payload is not None
    readiness_blockers = [
        *(
            []
            if inflate_parity_satisfied
            else ["candidate_requires_full_frame_inflate_output_parity_before_score_claim"]
        ),
        "candidate_requires_exact_auth_eval_before_promotion",
    ]
    entropy_position = classify_entropy_position(
        FECA_SELECTOR_TARGET_KIND,
        operation_family="selector_context_recode",
        payload_context={"entropy_position_id": "P11"},
    )
    upstream_positions = [str(item) for item in upstream_entropy_positions if str(item).strip()]
    downstream_targets = [
        str(item) for item in downstream_materializer_targets if str(item).strip()
    ]
    chain_composition = {
        "schema": "selector_context_recode_chain_composition.v1",
        "chain_label": chain_label,
        "chain_parent_artifact": str(chain_parent_artifact) if chain_parent_artifact else None,
        "upstream_entropy_positions": upstream_positions,
        "current_entropy_position": "P11",
        "current_materializer_target": FECA_SELECTOR_TARGET_KIND,
        "selected_codec_family": best_codec_family,
        "downstream_materializer_targets": downstream_targets,
        "downstream_entropy_positions": ["P15"] if downstream_targets else [],
        "composition_rule": "cascade_upstream_scorer_region_repairs_then_selector_context_recode_then_repack",
        "queue_owned_chain_required": True,
        "rate_only_stage": True,
        "distortion_budget_spend_allowed": False,
        "exact_axis_replay_required_before_budget_spend": True,
    }
    proof = apply_proxy_evidence_boundary(
        {
            "schema": FECA_REPARAMETERIZATION_PROOF_SCHEMA,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "target_kind": FECA_SELECTOR_TARGET_KIND,
            "materializer_id": FECA_SELECTOR_MATERIALIZER_ID,
            "operation_family": "selector_context_recode",
            "entropy_position_id": "P11",
            "entropy_position_classification": entropy_position,
            "receiver_contract_id": FECA_SELECTOR_RECEIVER_CONTRACT_ID,
            "receiver_contract_kind": FECA_SELECTOR_RECEIVER_CONTRACT_KIND,
            "source_archive": source_archive_record,
            "candidate_archive": candidate_archive_record,
            "candidate_runtime_dir": str(candidate_dir),
            "candidate_runtime_tree_sha256": candidate_runtime_tree_sha,
            "expected_runtime_tree_sha256": candidate_runtime_tree_sha,
            "runtime_adapter_ready": True,
            "candidate_runtime_adapter_blocker_cleared": True,
            "runtime_consumption_proof_passed": True,
            "passed": True,
            "receiver_contract_satisfied": True,
            "selector_code_roundtrip_equal": True,
            "source_payload_unchanged": True,
            "dqs1_tail_unchanged": True,
            "selected_codec_family": best_codec_family,
            "selected_context_order": _selector_codec_order(best_codec_family),
            "selected_context_model_kind": _selector_codec_model_kind(best_codec_family),
            "chain_composition": chain_composition,
            "selected_payload": {
                "source_archive_bytes": source_archive_record["bytes"],
                "candidate_archive_bytes": candidate_archive_record["bytes"],
                "saved_bytes": archive_saved_bytes,
                "source_payload_bytes": len(parts["selector_payload"]),
                "candidate_payload_bytes": len(best_payload),
                "payload_saved_bytes": len(parts["selector_payload"]) - len(best_payload),
                "status": "realized_saving",
                "savings_realized": archive_saved_bytes > 0,
            },
        },
        dispatch_blockers=tuple(readiness_blockers),
    )
    require_no_truthy_authority_fields(proof, context="feca_selector_runtime_consumption_proof")
    write_json_artifact(proof_path, proof)
    proof_record = _archive_record(proof_path)
    manifest = apply_proxy_evidence_boundary(
        {
            "schema": FECA_REPARAMETERIZATION_MANIFEST_SCHEMA,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "target_kind": FECA_SELECTOR_TARGET_KIND,
            "materializer_id": FECA_SELECTOR_MATERIALIZER_ID,
            "operation_family": "selector_context_recode",
            "entropy_position_id": "P11",
            "entropy_position_classification": entropy_position,
            "selection_scope": "fp11_feca_selector_stream",
            "receiver_contract_id": FECA_SELECTOR_RECEIVER_CONTRACT_ID,
            "receiver_contract_kind": FECA_SELECTOR_RECEIVER_CONTRACT_KIND,
            "portability_contract": "source_runtime_native_python_packet_decoder",
            "source_submission_dir": str(source_dir),
            "candidate_submission_dir": str(candidate_dir),
            "candidate_runtime_dir": str(candidate_dir),
            "candidate_runtime_tree_sha256": candidate_runtime_tree_sha,
            "expected_runtime_tree_sha256": candidate_runtime_tree_sha,
            "runtime_adapter_ready": True,
            "source_archive_path": str(source_archive),
            "source_archive": source_archive_record,
            "source_archive_bytes": source_archive_record["bytes"],
            "source_archive_sha256": source_archive_record["sha256"],
            "candidate_archive_path": str(candidate_archive),
            "candidate_archive": candidate_archive_record,
            "candidate_archive_bytes": candidate_archive_record["bytes"],
            "candidate_archive_sha256": candidate_archive_record["sha256"],
            "member_name": info.filename,
            "selected_member_name": info.filename,
            "selected_member_names": [info.filename],
            "source_member_bytes": len(member_payload),
            "candidate_member_bytes": len(candidate_member),
            "source_member": {
                "name": info.filename,
                "bytes": len(member_payload),
                "sha256": sha256_bytes(member_payload),
            },
            "candidate_member": {
                "name": info.filename,
                "bytes": len(candidate_member),
                "sha256": sha256_bytes(candidate_member),
            },
            "selector_source_bytes": len(parts["selector_payload"]),
            "selector_candidate_bytes": len(best_payload),
            "selector_saved_bytes": len(parts["selector_payload"]) - len(best_payload),
            "archive_saved_bytes": archive_saved_bytes,
            "selected_codec_family": best_codec_family,
            "selected_context_order": _selector_codec_order(best_codec_family),
            "selected_context_model_kind": _selector_codec_model_kind(best_codec_family),
            "selected_scale": best_scale,
            "selected_alpha": best_alpha,
            "requested_codec_families": [str(item) for item in codec_families],
            "codec_family_sweep_count": len(rows),
            "selector_code_count": len(source_codes),
            "selector_code_roundtrip_equal": True,
            "source_payload_unchanged": True,
            "dqs1_tail_unchanged": True,
            "candidate_runtime_patched": True,
            "byte_closed_candidate_emitted": True,
            "receiver_proof_ready": True,
            "receiver_contract_satisfied": True,
            "inflate_parity_satisfied": inflate_parity_satisfied,
            "full_frame_inflate_parity_proven": inflate_parity_satisfied,
            "full_frame_inflate_parity_source_path": (
                str(parity_source_path) if parity_source_path is not None else None
            ),
            "full_frame_inflate_parity_proof_path": (
                str(parity_copy_path) if parity_copy_record is not None else None
            ),
            "full_frame_inflate_parity_proof_bytes": (
                parity_copy_record["bytes"] if parity_copy_record is not None else None
            ),
            "full_frame_inflate_parity_proof_sha256": (
                parity_copy_record["sha256"] if parity_copy_record is not None else None
            ),
            "runtime_consumption_proof_path": str(proof_path),
            "runtime_consumption_proof_bytes": proof_record["bytes"],
            "runtime_consumption_proof_sha256": proof_record["sha256"],
            "receiver_verification": {
                "receiver_contract_satisfied": True,
                "runtime_adapter_ready": True,
                "candidate_runtime_dir": str(candidate_dir),
                "candidate_runtime_tree_sha256": candidate_runtime_tree_sha,
                "expected_runtime_tree_sha256": candidate_runtime_tree_sha,
                "blockers": [],
            },
            "candidate_runtime_patch": candidate_runtime_patch,
            "chain_composition": chain_composition,
            "selected_payload": {
                "source_archive_bytes": source_archive_record["bytes"],
                "candidate_archive_bytes": candidate_archive_record["bytes"],
                "saved_bytes": archive_saved_bytes,
                "source_payload_bytes": len(parts["selector_payload"]),
                "candidate_payload_bytes": len(best_payload),
                "payload_saved_bytes": len(parts["selector_payload"]) - len(best_payload),
                "status": "realized_saving",
                "savings_realized": archive_saved_bytes > 0,
            },
            "serialized_archive_delta": {
                "schema": "serialized_archive_delta_contract.v1",
                "source_archive_bytes": source_archive_record["bytes"],
                "candidate_archive_bytes": candidate_archive_record["bytes"],
                "saved_bytes": archive_saved_bytes,
            },
            "readiness_blockers": readiness_blockers,
            "sweep_rows": sorted(
                rows,
                key=lambda row: (
                    int(row.get("selector_payload_bytes") or 1 << 60),
                    str(row.get("codec_family") or ""),
                    int(row.get("scale") if row.get("scale") is not None else -1),
                    int(row.get("alpha") if row.get("alpha") is not None else -1),
                ),
            ),
        },
        dispatch_blockers=tuple(readiness_blockers),
    )
    require_no_truthy_authority_fields(manifest, context="feca_selector_reparameterization")
    write_json_artifact(output / "feca_selector_reparameterization_manifest.json", manifest)
    return manifest


__all__ = [
    "FECA_REPARAMETERIZATION_MANIFEST_SCHEMA",
    "FECA_REPARAMETERIZATION_PROOF_SCHEMA",
    "FECA_SELECTOR_MATERIALIZER_ID",
    "FECA_SELECTOR_RECEIVER_CONTRACT_ID",
    "FECA_SELECTOR_RECEIVER_CONTRACT_KIND",
    "FECA_SELECTOR_TARGET_KIND",
    "FecaSelectorReparameterizationError",
    "build_feca_selector_reparameterized_candidate",
    "join_fp11_member",
    "split_fp11_member",
]
