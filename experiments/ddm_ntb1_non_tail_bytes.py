"""Inventory and losslessly repackage move44 with its own shipped parsers.

Scorer-free research. Each deterministic format receives two real encodes;
all payloads are retained, including losing and intermediate representations.
No receiver mutation, scorer execution, seal or score claim is provided here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import lzma
import os
import shutil
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_ntb1_non_tail_bytes/reproduction")
BASE_SHA = "04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e"
AXIS = "[macOS-CPU advisory; exact bytes, scorer-free]"
EXCHANGE = 25 / 37545489


def fact(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def retain(path, payload):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("write outside owned store")
    if shutil.disk_usage(ROOT).free < 40 * 1024**3 + len(payload):
        raise RuntimeError("STORAGE_RESERVE: retain bytes; no deletion permitted")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable checkpoint drift: {path}")
    else:
        temp = path.with_name(path.name + ".partial")
        temp.write_bytes(payload)
        os.replace(temp, path)
    return fact(path)


def save(name, value):
    retain(ROOT / name, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
    return value


def inputs():
    runtime = ROOT / "source_runtime"
    if fact(runtime / "archive.zip")["sha256"] != BASE_SHA:
        raise ValueError("move44 source archive drift")
    rows = [
        fact(p)
        for p in sorted(runtime.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    ]
    value = {
        "archive": fact(runtime / "archive.zip"),
        "source_files": rows,
        "producer": fact(Path(__file__)),
        "seed": 20260911,
        "axis": AXIS,
        "score_claim": False,
        "n": 600,
        "retention": "All payloads retained on SSD; no disposable bulk. Refuse at 40 GiB reserve.",
        "upstream": fact(REPO / "upstream/evaluate.py"),
    }
    return save("INPUTS.json", value)


def load():
    inputs()
    r = ROOT / "source_runtime"
    sys.path[:0] = [str(r), str(r / "cpr1")]
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(20260911)
    torch.use_deterministic_algorithms(True)
    from runtime import residual_archive as rx

    spec = importlib.util.spec_from_file_location("ntb1_renderer", r / "cpr1/inflate.py")
    renderer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = renderer
    spec.loader.exec_module(renderer)
    return rx, renderer


def sections(rx):
    path = ROOT / "source_runtime/archive.zip"
    with zipfile.ZipFile(path) as z:
        if z.namelist() != ["p"]:
            raise ValueError("wrong member set")
        member = z.read("p")
    header = list(rx.RX1_MODEL_HEADER.unpack_from(member))
    offset = rx.RX1_MODEL_HEADER.size
    out = {"header": member[:offset]}
    for name, length in zip(("hpac", "semantic", "carrier"), header[5:], strict=True):
        out[name] = member[offset : offset + length]
        offset += length
    out["tail"] = member[offset:]
    for name, payload in out.items():
        retain(ROOT / "retained/source" / (name + ".bin"), payload)
    return out, header


def census():
    rx, renderer = load()
    from integer_model_io import deserialize_integer_model
    from runtime import ihs2
    from runtime import rc1_adaptive_model_sections as rc1
    from runtime import rc2_hpac_semistatic_mixing as rc2
    from runtime import rc3_shared_mixer as rc3
    from runtime import sm1_semantic_mixer as sm1

    out, header = sections(rx)
    parts = rx.read_residual_archive(ROOT / "source_runtime/archive.zip")
    template = renderer.SemanticTokenRenderer(renderer.SEMANTIC_WIDTH).state_dict()
    semantic = sm1.restore_semantic(parts.semantic_blob, template)
    retain(ROOT / "retained/semantic.sm3r", semantic)
    # The actual receiver walker drives every offset, shape and precision.
    offset, walk_fields = 10, []

    def read(kind, length):
        nonlocal offset
        payload = semantic[offset : offset + length]
        if len(payload) != length:
            raise ValueError("truncated semantic field")
        walk_fields.append({"kind": kind, "start": offset, "bytes": length})
        offset += length
        return payload

    plan = rc1.walk_sm3r(read, template, tuple(semantic[4:8]))
    if offset != len(semantic):
        raise ValueError("semantic census does not close")
    names = [name for name, t in template.items() if t.ndim >= 2]
    for row, item in zip(walk_fields, plan, strict=True):
        if item["kind"] == "codes":
            row.update(
                layer=names[item["group"]],
                bits=item["bits"],
                count=item["count"],
                padding_bits=item["length"] * 8 - item["count"] * item["bits"],
            )
    layout = ihs2.layout_from_runtime(renderer)
    hpac = rx.materialize_ihs1(parts.hpac_blob, renderer)
    retain(ROOT / "retained/hpac.ihs1", hpac)
    # Execute the shipping integer loader, not just the structural walker.
    model = renderer.load_hpac(hpac, __import__("torch").device("cpu"))
    deserialize_integer_model(model, hpac)
    prefix, packed, tail, depths = rc2.split_ihs1(hpac, list(layout.row_counts))
    hpac_fields = []
    for name, start, stop in layout.module_ranges:
        hpac_fields.append(
            {
                "layer": name,
                "rows": stop - start,
                "depths": depths[start:stop].tolist(),
                "weight_bits": sum(int(depths[i]) * layout.row_counts[i] for i in range(start, stop)),
            }
        )
    for label, payload in [
        ("hpac_prefix", prefix),
        ("hpac_weights", packed),
        ("hpac_tail", tail),
        ("semantic_rider", parts.semantic_blob),
        ("hpac_rider", parts.hpac_blob),
        ("carrier_decoded", parts.carrier_blob),
        ("token_stream", parts.token_stream),
    ]:
        retain(ROOT / "retained" / (label + ".bin"), payload)
    rh = rc3.HEADER.unpack_from(parts.hpac_blob)
    sh = sm1.HEADER.unpack_from(parts.semantic_blob)
    with zipfile.ZipFile(ROOT / "source_runtime/archive.zip") as z:
        zi = z.infolist()[0]
        zip_census = {
            "local_fixed": 30,
            "local_name": len(zi.filename.encode()),
            "local_extra": len(zi.extra),
            "central_fixed": 46,
            "central_name": len(zi.filename.encode()),
            "central_extra": len(zi.extra),
            "member_comment": len(zi.comment),
            "eocd_fixed": 22,
            "archive_comment": len(z.comment),
            "compression_method": zi.compress_type,
        }
    non_tail = 100 + sum(len(out[k]) for k in ("header", "hpac", "semantic", "carrier"))
    if non_tail != 60497 or sum(map(len, out.values())) + 100 != 180406:
        raise ValueError("non-tail does not reconcile to move44")
    result = {
        "axis": AXIS,
        "score_claim": False,
        "archive": fact(ROOT / "source_runtime/archive.zip"),
        "container": {k: len(v) for k, v in out.items()},
        "non_tail_bytes": non_tail,
        "zip": zip_census,
        "tail": {
            "fixed_table": 96,
            "counted_rider": len(out["tail"]) - 96 - len(parts.token_stream),
            "arithmetic_stream": len(parts.token_stream),
        },
        "semantic": {
            "body_bytes": len(semantic),
            "header_and_selection": 10,
            "fields": walk_fields,
            "rider_header": sm1.HEADER.size,
            "metadata": len(parts.semantic_blob) - sm1.HEADER.size - 24 - sh[-1],
            "mixer_weights": 24,
            "coded_weights": sh[-1],
        },
        "hpac": {
            "body_bytes": len(hpac),
            "magic": 4,
            "depth_table": len(prefix) - 4,
            "packed_weight_bytes": len(packed),
            "padding_bits": len(packed) * 8 - sum(r["weight_bits"] for r in hpac_fields),
            "fields": hpac_fields,
            "tail_fields": [
                {"name": f.name, "bytes": f.byte_count, "dtype": f.dtype, "shape": f.shape} for f in layout.tail_fields
            ],
            "rider_header": rc3.HEADER.size,
            "prefix": rh[-3],
            "parameters": 25,
            "coded_weights": rh[-2],
            "tail": rh[-1],
        },
        "attribution": "Layer counts are restored body bytes/bits. Shared arithmetic coding and outer Brotli prevent additive compressed layer attribution.",
        "wans_parser": "runtime/entropy/renderer_weight_codec.py inspected; WANS branch is not used by shipped SM1S/SM3R body.",
        "receiver_models_loaded": True,
    }
    return save("CENSUS.json", result)


def pack(member, method, path):
    # Two independent real ZipFile encodes, no timestamp/seed search.
    import io

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=method) as z:
        zi = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = method
        zi.external_attr = 0o600 << 16
        z.writestr(zi, member)
    payload = stream.getvalue()
    retain(path, payload)
    with zipfile.ZipFile(path) as z:
        if z.namelist() != ["p"] or z.read("p") != member or z.testzip() is not None:
            raise ValueError("ZIP parse-back changed payload")
    return payload


def price():
    rx, _ = load()
    import brotli

    out, header = sections(rx)
    base = rx.read_residual_archive(ROOT / "source_runtime/archive.zip")
    rows = []
    # Fixed format roster, no entropy-model fitting or tail coder race.
    variants = [
        ("zip_stored", "zip", zipfile.ZIP_STORED),
        ("zip_deflate", "zip", zipfile.ZIP_DEFLATED),
        ("zip_bzip2", "zip", zipfile.ZIP_BZIP2),
        ("zip_lzma", "zip", zipfile.ZIP_LZMA),
    ]
    for key in ("hpac", "semantic", "carrier"):
        variants.extend([(key + "_brotli_q11", key, "q11"), (key + "_brotli_q10", key, "q10")])
    variants.extend(
        [
            ("hpac_xz", "hpac", "xz"),
            ("semantic_plane_toggle", "semantic", "plane"),
            ("carrier_plane_toggle", "carrier", "plane"),
        ]
    )
    for name, key, method in variants:
        target = ROOT / "retained/prices" / name
        twins = []
        for twin in range(2):
            h, changed = header.copy(), out.copy()
            zip_method = method if key == "zip" else zipfile.ZIP_STORED
            if key != "zip":
                body = brotli.decompress(out[key])
                if method == "plane":
                    span = len(body) & ~1
                    # Toggle only the already-admitted CK2 representation.
                    bit = 2 if key == "semantic" else 4
                    body = (
                        rx._ck2_uninterleave_planes(body)
                        if h[4] & bit
                        else body[:span:2] + body[1:span:2] + body[span:]
                    )
                    h[4] ^= bit
                retain(target / f"body.twin{twin}.bin", body)
                compressed = (
                    lzma.compress(body, format=lzma.FORMAT_XZ, preset=9)
                    if method == "xz"
                    else brotli.compress(body, quality=10 if method == "q10" else 11, lgwin=24)
                )
                retain(target / f"section.twin{twin}.bin", compressed)
                changed[key] = compressed
                h[5 + ("hpac", "semantic", "carrier").index(key)] = len(compressed)
                if method == "xz":
                    h[2] = 1
                changed["header"] = rx.RX1_MODEL_HEADER.pack(*h)
            member = b"".join(changed[k] for k in ("header", "hpac", "semantic", "carrier", "tail"))
            retain(target / f"member.twin{twin}.bin", member)
            payload = pack(member, zip_method, target / f"archive.twin{twin}.zip")
            parsed = rx.read_residual_archive(target / f"archive.twin{twin}.zip")
            for field in (
                "semantic_blob",
                "carrier_blob",
                "hpac_blob",
                "residual_payload",
                "token_stream",
                "tc1_weights",
                "compensation_blob",
            ):
                if getattr(base, field) != getattr(parsed, field):
                    raise ValueError(f"{name}: receiver field changed: {field}")
            twins.append(payload)
        if twins[0] != twins[1]:
            raise ValueError("deterministic twins disagree; retain and investigate")
        delta = len(twins[0]) - 180406
        row = {
            "lever": name,
            "archive": fact(target / "archive.twin0.zip"),
            "delta_bytes": delta,
            "delta_s_conditional": delta * EXCHANGE,
            "delta_d_seg_derived": 0,
            "delta_d_pose_derived": 0,
            "n600_scorer_measured": False,
            "parsed_model_and_tail_identity": True,
            "twins_identical": True,
            "clears_bar": delta * EXCHANGE <= -2e-5,
            "axis": AXIS,
            "score_claim": False,
        }
        save(f"retained/prices/{name}/RESULT.json", row)
        rows.append(row)
        print(json.dumps(row), flush=True)
    return save(
        "PRICES.json",
        {
            "rows": rows,
            "score_claim": False,
            "axis": AXIS,
            "best": min(rows, key=lambda r: r["archive"]["bytes"]),
            "renderer_resolve": "QUEUED: no scorer ownership assigned by charter/common contract",
            "hpac_lossy": "UNMEASURED: requires move44 RLC1-aware causal tail re-encode; older JG2 encoder is not that mechanism",
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["census", "price"])
    p.add_argument("--resume-from", type=Path, required=True)
    a = p.parse_args()
    if a.resume_from.resolve() != ROOT.resolve():
        raise ValueError("wrong resume root")
    path = ROOT / ("CENSUS.json" if a.stage == "census" else "PRICES.json")
    inputs()
    if path.exists():
        print(json.dumps({"stage_complete": fact(path)}))
        return
    {"census": census, "price": price}[a.stage]()
    print(json.dumps({"complete": a.stage, "receipt": fact(path), "score_claim": False}), flush=True)


if __name__ == "__main__":
    main()
