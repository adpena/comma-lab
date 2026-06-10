# SPDX-License-Identifier: MIT
"""Lossless per-tensor entropy recode of an FP11 (PR #101/#110) frontier packet.

This is the canonical materializer for the ``pr110_payload_entropy_recode`` lane
(``.omx/research/public_pr112_frontier_beat_intake_20260610.md`` MOVE 1 + MOVE 2):
it applies PR #112's per-tensor adaptive range coder + per-dim AR latent coder
(``tac.packet_compiler.ctx_range_coder``) to the **decoder-weight** and **latent**
sections of an FP11 packet, replacing PR #101's split-Brotli decoder blob and
raw-LZMA1 latent blob with a tighter entropy coding. The recode is **lossless by
construction**: the reconstructed raw decoder streams + raw latent payload are
byte-identical to the originals, so the decoded pixels (and therefore d_seg /
d_pose) are unchanged. The entire win is on the rate axis.

The FECa/FEC8 selector payload and the optional DQS1 tail are kept **verbatim**:
PR #112's selector recoder only understands the plain ``FEC6`` magic, whereas our
frontier packets carry the richer ``FECa`` (FEC10-hybrid) selector that spent
extra bytes to win on the distortion axes. Re-coding it would be a regression
(and a parse-back hazard), so it is preserved bit-for-bit.

Container grammar of the recoded source payload (replaces the legacy
``[brotli decoder | lzma latents | sidecar]`` source payload):

    CTXR magic (4) | u8 version | u24 dec_sec_len | u24 lat_sec_len
        | u24 sidecar_len | dec_sec | lat_sec | sidecar

``dec_sec`` and ``lat_sec`` are ``ctx_range_coder`` blobs; ``sidecar`` is the
607-byte PR #101 canonical-Huffman latent-correction sidecar, verbatim.

Reuses:
  - ``tac.packet_compiler.feca_selector_reparameterize.split_fp11_member`` /
    ``join_fp11_member`` for the FP11 outer wrapper + selector + DQS1 tail.
  - ``tac.packet_compiler.ctx_range_coder`` (the absorbed PR #112 coder).
The brotli stream splitting + LZMA filter constants are PR #101 schema constants
re-declared here so this module does not import a per-submission ``codec.py``.

Inflate-side deps: numpy + constriction (harness base env). Build-side adds brotli
+ lzma (stdlib) for decompressing the legacy source sections.
"""
from __future__ import annotations

import lzma
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tac.packet_compiler import ctx_range_coder
from tac.packet_compiler.feca_selector_reparameterize import (
    FEC8_MAGIC,
    FECA_MAGIC,
    join_fp11_member,
    split_fp11_member,
)

# --- PR #101 source-payload schema (verified against R3 submission_dir codec.py) ---
# DECODER_BLOB_LEN is the *Brotli-compressed* decoder blob length; it is packet
# specific (R3 = 162127, PR #110 = 162164) and is read from the packet's codec.py
# when available, else inferred from the source payload length minus the constant
# latent + sidecar lengths.
LATENT_BLOB_LEN = 15_387  # PR #101 raw-LZMA1 latent blob (constant)
SIDECAR_LEN = 607  # PR #101 canonical-Huffman latent-correction sidecar (constant)
N_DECODER_STREAMS = 7  # PR #101 split-Brotli decoder stream count (STREAM_ENDS)
LATENT_LZMA_FILTERS = [
    {"id": lzma.FILTER_LZMA1, "dict_size": 4096, "lc": 3, "lp": 0, "pb": 0}
]

CTXR_MAGIC = b"CTXR"
CTXR_VERSION = 1
CTXR_HEADER_LEN = 4 + 1 + 3 + 3 + 3  # magic | version | 3x u24 lengths

PR110_PAYLOAD_ENTROPY_RECODE_TARGET_KIND = "pr110_payload_entropy_recode_v1"
PR110_PAYLOAD_ENTROPY_RECODE_MATERIALIZER_ID = "pr110_payload_entropy_recode_adapter"


class Pr110PayloadEntropyRecodeError(ValueError):
    """Raised when a payload entropy recode is unsafe or non-lossless."""


# ---------------------------------------------------------------------------
# brotli stream split (PR #101 split-Brotli decoder blob -> per-stream raws)
# ---------------------------------------------------------------------------
def decompress_brotli_streams_to_list(data: bytes, n_streams: int) -> list[bytes]:
    """Decompress ``n_streams`` independent Brotli streams; return the list.

    Mirrors PR #101's ``decompress_brotli_streams`` but returns the per-stream
    list (not joined), which is the input contract of ``ctx_range_coder``.
    """
    import brotli  # build-side only

    outputs: list[bytes] = []
    pos = 0
    for _ in range(n_streams):
        dec = brotli.Decompressor()
        chunks: list[bytes] = []
        while pos < len(data) and not dec.is_finished():
            chunks.append(dec.process(data[pos : pos + 1]))
            pos += 1
        if not dec.is_finished():
            raise Pr110PayloadEntropyRecodeError("truncated split-Brotli decoder payload")
        outputs.append(b"".join(chunks))
    if pos != len(data):
        raise Pr110PayloadEntropyRecodeError("trailing split-Brotli decoder payload")
    return outputs


# ---------------------------------------------------------------------------
# CTXR container pack / unpack
# ---------------------------------------------------------------------------
def pack_ctxr_container(dec_sec: bytes, lat_sec: bytes, sidecar: bytes) -> bytes:
    if len(dec_sec) > 0xFFFFFF or len(lat_sec) > 0xFFFFFF or len(sidecar) > 0xFFFFFF:
        raise Pr110PayloadEntropyRecodeError("CTXR section length exceeds u24")
    return (
        CTXR_MAGIC
        + bytes([CTXR_VERSION])
        + len(dec_sec).to_bytes(3, "little")
        + len(lat_sec).to_bytes(3, "little")
        + len(sidecar).to_bytes(3, "little")
        + dec_sec
        + lat_sec
        + sidecar
    )


def unpack_ctxr_container(blob: bytes) -> tuple[bytes, bytes, bytes]:
    """-> (dec_sec, lat_sec, sidecar). Validates magic + version + lengths."""
    if len(blob) < CTXR_HEADER_LEN:
        raise Pr110PayloadEntropyRecodeError("CTXR container too short")
    if blob[:4] != CTXR_MAGIC:
        raise Pr110PayloadEntropyRecodeError(f"bad CTXR magic: {blob[:4]!r}")
    if blob[4] != CTXR_VERSION:
        raise Pr110PayloadEntropyRecodeError(f"unsupported CTXR version {blob[4]}")
    p = 5
    dl = int.from_bytes(blob[p : p + 3], "little")
    p += 3
    ll = int.from_bytes(blob[p : p + 3], "little")
    p += 3
    sl = int.from_bytes(blob[p : p + 3], "little")
    p += 3
    dec_sec = blob[p : p + dl]
    p += dl
    lat_sec = blob[p : p + ll]
    p += ll
    sidecar = blob[p : p + sl]
    p += sl
    if p != len(blob):
        raise Pr110PayloadEntropyRecodeError("CTXR container has trailing bytes")
    if len(dec_sec) != dl or len(lat_sec) != ll or len(sidecar) != sl:
        raise Pr110PayloadEntropyRecodeError("CTXR section length mismatch")
    return dec_sec, lat_sec, sidecar


# ---------------------------------------------------------------------------
# decoder/latent recode (encode) + reconstruction (decode)
# ---------------------------------------------------------------------------
def encode_source_payload(
    source_payload: bytes, *, decoder_blob_len: int
) -> tuple[bytes, dict[str, Any]]:
    """Recode the legacy ``[brotli decoder | lzma latents | sidecar]`` source
    payload into a CTXR container; return (ctxr_container, metrics)."""
    if ctx_range_coder.constriction is None:  # pragma: no cover
        raise Pr110PayloadEntropyRecodeError("constriction is required for the recode")
    expected = decoder_blob_len + LATENT_BLOB_LEN + SIDECAR_LEN
    if len(source_payload) != expected:
        raise Pr110PayloadEntropyRecodeError(
            f"source payload len {len(source_payload)} != "
            f"decoder({decoder_blob_len})+latent({LATENT_BLOB_LEN})+sidecar({SIDECAR_LEN})="
            f"{expected}"
        )
    decoder_blob = source_payload[:decoder_blob_len]
    latent_blob = source_payload[decoder_blob_len : decoder_blob_len + LATENT_BLOB_LEN]
    sidecar = source_payload[decoder_blob_len + LATENT_BLOB_LEN :]

    raw_streams = decompress_brotli_streams_to_list(decoder_blob, N_DECODER_STREAMS)
    dec_sec = ctx_range_coder.encode_decoder_section(raw_streams)

    latent_raw = lzma.decompress(
        latent_blob, format=lzma.FORMAT_RAW, filters=LATENT_LZMA_FILTERS
    )
    lat_sec = ctx_range_coder.encode_latent_section(latent_raw)

    container = pack_ctxr_container(dec_sec, lat_sec, sidecar)
    metrics = {
        "decoder_blob_len": decoder_blob_len,
        "latent_blob_len": LATENT_BLOB_LEN,
        "sidecar_len": SIDECAR_LEN,
        "dec_sec_bytes": len(dec_sec),
        "lat_sec_bytes": len(lat_sec),
        "ctxr_header_bytes": CTXR_HEADER_LEN,
        "decoder_delta_bytes": len(dec_sec) - decoder_blob_len,
        "latent_delta_bytes": len(lat_sec) - LATENT_BLOB_LEN,
        "source_payload_delta_bytes": len(container) - len(source_payload),
        "orig_source_payload_bytes": len(source_payload),
        "recoded_source_payload_bytes": len(container),
    }
    return container, metrics


def reconstruct_raw_sections(ctxr_container: bytes) -> tuple[list[bytes], bytes, bytes]:
    """Decode a CTXR container into (raw_decoder_streams, raw_latent_payload, sidecar).

    The runtime feeds ``b"".join(raw_decoder_streams)`` to the PR #101 reshape and
    ``raw_latent_payload`` to the post-LZMA latent reshape, byte-exactly.
    """
    dec_sec, lat_sec, sidecar = unpack_ctxr_container(ctxr_container)
    raw_streams = ctx_range_coder.decode_decoder_section(dec_sec)
    latent_raw = ctx_range_coder.decode_latent_section(lat_sec)
    return raw_streams, latent_raw, sidecar


def verify_lossless(source_payload: bytes, *, decoder_blob_len: int) -> dict[str, Any]:
    """Encode then decode; assert raw decoder streams + latent payload + sidecar
    are byte-identical to the originals. Raises on any mismatch.

    This is the fail-closed lossless gate: it proves the recode reproduces the
    exact bytes the PR #101 reshape consumes, so the decoded pixels are identical.
    """
    decoder_blob = source_payload[:decoder_blob_len]
    latent_blob = source_payload[decoder_blob_len : decoder_blob_len + LATENT_BLOB_LEN]
    sidecar = source_payload[decoder_blob_len + LATENT_BLOB_LEN :]

    orig_streams = decompress_brotli_streams_to_list(decoder_blob, N_DECODER_STREAMS)
    orig_joined = b"".join(orig_streams)
    orig_latent_raw = lzma.decompress(
        latent_blob, format=lzma.FORMAT_RAW, filters=LATENT_LZMA_FILTERS
    )

    container, _ = encode_source_payload(source_payload, decoder_blob_len=decoder_blob_len)
    rt_streams, rt_latent_raw, rt_sidecar = reconstruct_raw_sections(container)
    rt_joined = b"".join(rt_streams)

    decoder_ok = rt_joined == orig_joined
    latent_ok = rt_latent_raw == orig_latent_raw
    sidecar_ok = rt_sidecar == sidecar
    if not (decoder_ok and latent_ok and sidecar_ok):
        raise Pr110PayloadEntropyRecodeError(
            "non-lossless recode: "
            f"decoder_ok={decoder_ok} latent_ok={latent_ok} sidecar_ok={sidecar_ok}"
        )
    return {
        "decoder_raw_byte_identical": decoder_ok,
        "latent_raw_byte_identical": latent_ok,
        "sidecar_byte_identical": sidecar_ok,
        "decoder_raw_sha256": _sha256(orig_joined),
        "latent_raw_sha256": _sha256(orig_latent_raw),
        "sidecar_sha256": _sha256(sidecar),
    }


# ---------------------------------------------------------------------------
# member-level recode
# ---------------------------------------------------------------------------
@dataclass
class RecodeResult:
    orig_member: bytes
    recoded_member: bytes
    selector_payload: bytes
    dqs1_tail: bytes
    decoder_blob_len: int
    metrics: dict[str, Any] = field(default_factory=dict)
    lossless_proof: dict[str, Any] = field(default_factory=dict)

    @property
    def orig_member_bytes(self) -> int:
        return len(self.orig_member)

    @property
    def recoded_member_bytes(self) -> int:
        return len(self.recoded_member)

    @property
    def member_delta_bytes(self) -> int:
        return len(self.recoded_member) - len(self.orig_member)


def recode_fp11_member(member_bytes: bytes, *, decoder_blob_len: int) -> RecodeResult:
    """Recode an FP11 member's source payload; keep selector + DQS1 tail verbatim.

    Always verifies losslessness before returning (fail-closed).
    """
    parts = split_fp11_member(
        member_bytes, allowed_selector_magics=(FECA_MAGIC, FEC8_MAGIC)
    )
    source_payload = parts["source_payload"]
    selector_payload = parts["selector_payload"]
    dqs1_tail = parts["dqs1_tail"]

    lossless_proof = verify_lossless(source_payload, decoder_blob_len=decoder_blob_len)
    container, metrics = encode_source_payload(
        source_payload, decoder_blob_len=decoder_blob_len
    )
    recoded_member = join_fp11_member(
        source_payload=container,
        selector_payload=selector_payload,
        dqs1_tail=dqs1_tail,
    )
    # round-trip safety: the recoded member must split back to the same selector +
    # DQS1 tail, and its source must decode to the same raw sections.
    rt_parts = split_fp11_member(
        recoded_member, allowed_selector_magics=(FECA_MAGIC, FEC8_MAGIC)
    )
    if rt_parts["selector_payload"] != selector_payload:
        raise Pr110PayloadEntropyRecodeError("recoded member changed selector payload")
    if rt_parts["dqs1_tail"] != dqs1_tail:
        raise Pr110PayloadEntropyRecodeError("recoded member changed DQS1 tail")
    rt_streams, rt_latent_raw, rt_sidecar = reconstruct_raw_sections(
        rt_parts["source_payload"]
    )
    orig_streams = decompress_brotli_streams_to_list(
        source_payload[:decoder_blob_len], N_DECODER_STREAMS
    )
    if b"".join(rt_streams) != b"".join(orig_streams):
        raise Pr110PayloadEntropyRecodeError("recoded member decoder raw mismatch")
    orig_latent_raw = lzma.decompress(
        source_payload[decoder_blob_len : decoder_blob_len + LATENT_BLOB_LEN],
        format=lzma.FORMAT_RAW,
        filters=LATENT_LZMA_FILTERS,
    )
    if rt_latent_raw != orig_latent_raw:
        raise Pr110PayloadEntropyRecodeError("recoded member latent raw mismatch")
    if rt_sidecar != source_payload[decoder_blob_len + LATENT_BLOB_LEN :]:
        raise Pr110PayloadEntropyRecodeError("recoded member sidecar mismatch")

    return RecodeResult(
        orig_member=member_bytes,
        recoded_member=recoded_member,
        selector_payload=selector_payload,
        dqs1_tail=dqs1_tail,
        decoder_blob_len=decoder_blob_len,
        metrics=metrics,
        lossless_proof=lossless_proof,
    )


# ---------------------------------------------------------------------------
# archive helpers
# ---------------------------------------------------------------------------
def read_single_member(archive_path: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(archive_path) as zf:
        infos = zf.infolist()
        if len(infos) != 1:
            raise Pr110PayloadEntropyRecodeError(
                f"expected one archive member, found {len(infos)}"
            )
        return infos[0].filename, zf.read(infos[0].filename)


def write_stored_archive(archive_path: Path, *, member_name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(member_name)
    info.compress_type = zipfile.ZIP_STORED
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.external_attr = 0
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(info, payload)


def infer_decoder_blob_len(source_submission_dir: Path) -> int:
    """Read DECODER_BLOB_LEN from the packet's ``src/codec.py`` (the source-of-truth)."""
    codec_path = source_submission_dir / "src" / "codec.py"
    if not codec_path.exists():
        raise Pr110PayloadEntropyRecodeError(f"missing {codec_path}")
    for line in codec_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("DECODER_BLOB_LEN"):
            value = stripped.split("=", 1)[1].strip()
            return int(value.replace("_", ""))
    raise Pr110PayloadEntropyRecodeError("DECODER_BLOB_LEN not found in codec.py")


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()
