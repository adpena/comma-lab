"""ddm_ren2 deliverable 1: RESTORE the renderer warm-start init and prove the step-0 control.

`ddm_ren1` §3A measured that the only renderer warm-start init on either tier
(`ddm_ft1`'s `init_shipped_semantic_renderer.pt`) carries the 38 weight tensors and
nothing else that the DEPLOYED object needs: the per-tensor bit-depth table collapses
to the scalar `quant_bits: 4` (a UNIFORM grid), `keep_percent` and the FiLM row-prune
mask are absent, the per-axis fp16 scale rule and its 5.96e-08 floor are re-derived by
whatever the trainer happens to do, and `provenance.archive_sha256` binds an archive
that is three moves stale.  That is the measured cause of `ddm_ft1` refitting a
DIFFERENT object (uniform-int4, unpruned) and reporting +31.23 % as if it were a
verdict on the deployed renderer.

This producer does the opposite of inferring any of that from a flag.  It reads the
deployed representation OUT OF THE SHIPPED BYTES (`rp1` r2: *"prove the realized
quantizer from the packed bytes at step 0, not from the flags"*), writes a restored
init that carries it, and then proves the restoration is exact by re-encoding the
restored state through the REAL chain and demanding byte identity with the shipped
member at all three stages:

    pack_prune_mixed_candidate -> SM3R body   (36,130 B, sha 17e0fd0b...)
    sm1_semantic_mixer.encode  -> SM1S rider  (31,451 B, sha 9c0b579c...)
    CK2 interleave + brotli    -> RX1M member (29,862 B, sha 786950a5...)

Byte identity at the member is what makes this producer the arm's member PRICER: a
candidate's member delta is measured by the same three calls, so the delta is a real
encode and never an extrapolation of `ddm_fe1`'s +70 B break fee (which was measured
at N <= 200 changed codes; a refit changes all 59,376).

It also writes the training conditioning cache from the CURRENT coded field
`a92e7d90...`, which is the rung itself: the shipped renderer was trained on GT-oracle
tokens and has never been fit to the field it actually receives.

Axis: `[macOS-CPU advisory, byte-exact encoder control]`.  No score claim, no
promotion, no archive is built, no scorer is run here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "src")]

import numpy as np

AXIS = "[macOS-CPU advisory, byte-exact encoder control]"

#: Move 48 -- the pointer this arm is bound to.  Read-only, never written.
POINTER_TREE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_hpr1/public/retrain_frame_even/candidate_runtime"
)
POINTER_ARCHIVE_SHA256 = (
    "d830edd371641e1968765ae6be27120a6c55a9b1ca4b3747158e603a43ef149c"
)
POINTER_ARCHIVE_BYTES = 179_111

#: The shipped coded token field -- the conditioning input the refit is FOR.
TOKEN_FIELD = Path("/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/subset6.u8")
TOKEN_FIELD_SHA256 = (
    "a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8"
)

#: `ddm_ft1`'s warm-start init -- the object whose dropped state this producer restores.
FT1_INIT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_ft1_shipped_renderer_aligned_finetune/retained"
    "/init_shipped_semantic_renderer.pt"
)
#: `ddm_ft1`'s DALI GT seg target cache.  Verified here to be byte-identical to
#: `jg1.load_gt_seg_labels(LINEAGE_DALI)`, so it is reused rather than rebuilt.
FT1_TARGET_CACHE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_ft1_shipped_renderer_aligned_finetune/retained"
    "/cache_target_dali_seg.pt"
)

STORE = Path("/Volumes/VertigoDataTier/pact/ddm_ren2")
RESERVE_BYTES = 40 << 30

#: The shipped member chain, pinned by BYTES so a drift refuses instead of re-baselining.
SHIPPED_BODY_BYTES = 36_130
SHIPPED_BODY_SHA256 = (
    "17e0fd0b197ac147afe98397ef38f02f7915b69372d03c042e6be6fa0f992e50"
)
SHIPPED_RIDER_BYTES = 31_451
SHIPPED_MEMBER_BYTES = 29_862

#: Candidate Brotli containers searched at the control.  The shipped one is IDENTIFIED
#: by byte identity, never assumed: `ddm_fe1` is cited upstream as pinning the shipped
#: semantic shape at (ck2, q11, lgwin24) and that is NOT what this archive carries.
BROTLI_GRID = tuple((q, lg) for q in (9, 10, 11) for lg in (16, 18, 20, 22, 24))


class Ren2InitError(RuntimeError):
    """Refusal raised by this producer.  Never downgraded to a warning."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def retain(path: Path, payload: bytes) -> dict[str, Any]:
    """Persist a payload inside this arm's store, refusing to fill the tier."""
    if not path.resolve().is_relative_to(STORE.resolve()):
        raise Ren2InitError(f"write outside the ren2 store: {path}")
    if shutil.disk_usage(STORE.parent).free < RESERVE_BYTES + len(payload):
        raise Ren2InitError("STORAGE_BLOCK: keep all payloads; free space below reserve")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return fact(path)


def _receiver_modules():
    """Import the SHIPPED receiver out of the pointer tree, never a look-alike.

    `runtime/residual_archive.py` uses relative imports, so the package parent has to be
    on the path exactly as the shipped `inflate.py` puts it there.  Loading a hand-rolled
    re-parse instead would be the duplication the up3 canonical-helper lesson forbids.
    """
    import importlib

    added = [str(POINTER_TREE), str(POINTER_TREE / "cpr1")]
    for entry in added:
        if entry not in sys.path:
            sys.path.insert(0, entry)
    residual = importlib.import_module("runtime.residual_archive")
    mixer = importlib.import_module("runtime.sm1_semantic_mixer")
    inflate = importlib.import_module("inflate")
    receiver = importlib.import_module("ddm_mp2_semantic_receiver")
    return residual, mixer, inflate, receiver


def bind_pointer() -> dict[str, Any]:
    """Fail closed unless the archive and the token field are the bytes claimed."""
    archive = POINTER_TREE / "archive.zip"
    observed = sha256_file(archive)
    if observed != POINTER_ARCHIVE_SHA256:
        raise Ren2InitError(f"pointer archive drifted: {observed}")
    if archive.stat().st_size != POINTER_ARCHIVE_BYTES:
        raise Ren2InitError("pointer archive size drifted")
    if sha256_file(TOKEN_FIELD) != TOKEN_FIELD_SHA256:
        raise Ren2InitError("coded token field drifted")
    return {"archive": fact(archive), "token_field": fact(TOKEN_FIELD)}


def read_shipped_semantic_container() -> dict[str, Any]:
    """Slice the RX1M container and return every layer of the shipped semantic section.

    Returns the compressed member stream, the staged (post-CK2) bytes, the SM1S rider,
    the SM3R body, and the container flags -- all MEASURED off the archive under test.
    """
    import zipfile

    residual, mixer, inflate, receiver = _receiver_modules()
    with zipfile.ZipFile(POINTER_TREE / "archive.zip") as archive:
        names = archive.namelist()
        if names != ["p"]:
            raise Ren2InitError(f"pointer archive members are {names}, expected ['p']")
        outer = archive.read("p")
    header = residual.RX1_MODEL_HEADER
    if len(outer) < header.size or not outer.startswith(residual.RX1_MAGIC):
        raise Ren2InitError("pointer archive is not the RX1M container")
    (
        magic,
        version,
        codec,
        table_mode,
        reserved,
        hpac_bytes,
        semantic_bytes,
        carrier_bytes,
    ) = header.unpack_from(outer)
    if codec != residual.RX1_CODEC_BROTLI:
        raise Ren2InitError("the semantic member is not the Brotli RX1M codec")
    offset = header.size + hpac_bytes
    member = outer[offset : offset + semantic_bytes]
    staged = residual._decompress_brotli(member)
    rider = staged
    if reserved & residual.SZ1_RESERVED_SEMANTIC_SPLIT:
        rider = residual._sz1_unsplit_semantic(rider)
    ck2 = bool(reserved & residual.CK2_RESERVED_SEMANTIC_PLANE2)
    if ck2:
        rider = residual._ck2_uninterleave_planes(rider)
    if not rider.startswith(mixer.MAGIC):
        raise Ren2InitError("the shipped semantic rider is not SM1S")
    template = inflate.SemanticTokenRenderer(96).state_dict()
    body = receiver.restore_sm1_semantic(rider, template)
    return {
        "rx1_header": [
            magic.decode("ascii"),
            int(version),
            int(codec),
            int(table_mode),
            int(reserved),
            int(hpac_bytes),
            int(semantic_bytes),
            int(carrier_bytes),
        ],
        "ck2_semantic": ck2,
        "sz1_split": bool(reserved & residual.SZ1_RESERVED_SEMANTIC_SPLIT),
        "member": member,
        "staged": staged,
        "rider": rider,
        "body": body,
        "template": template,
    }


def measure_deployed_representation(body: bytes, template) -> dict[str, Any]:
    """Read the depth table, keep_percent and prune geometry OUT OF THE PACKED BYTES."""
    import ddm_sd1_semantic_rd_curve as sd1
    import ddm_sm3_semantic_representation as sm3

    if len(body) < 10 or body[:4] != sm3.MAGIC:
        raise Ren2InitError("the shipped semantic body is not SM3R")
    version, mode, keep_percent, reserved = body[4], body[5], body[6], body[7]
    if version != sm3.VERSION or mode != sm3.MODE_ROW_PRUNE_MIXED or reserved != 0:
        raise Ren2InitError(
            f"shipped SM3R header is (v{version}, mode {mode}, reserved {reserved}); "
            f"expected (v{sm3.VERSION}, mode {sm3.MODE_ROW_PRUNE_MIXED}, reserved 0)"
        )
    qnames = sd1.quantized_names(template)
    mask = struct.unpack_from("<H", body, 8)[0]
    expected_mask = sm3.mask_for_names(qnames, sm3.PRUNE_NAMES)
    if mask != expected_mask:
        raise Ren2InitError(
            f"shipped prune selection mask {mask} != PRUNE_NAMES mask {expected_mask}"
        )
    nibble_bytes = (len(qnames) + 1) // 2
    depths = sd1._unpack_depth_nibbles(body[10 : 10 + nibble_bytes], len(qnames))
    allocation = {name: int(bits) for name, bits in zip(qnames, depths, strict=True)}
    return {
        "mode": int(mode),
        "mode_name": "MODE_ROW_PRUNE_MIXED",
        "keep_percent": int(keep_percent),
        "prune_names": sorted(sm3.PRUNE_NAMES),
        "selection_mask": int(mask),
        "bit_allocation": allocation,
        "low_bit_names": sorted(name for name, bits in allocation.items() if bits == 3),
        "high_bits": 4,
        "low_bits": 3,
        "scale_rule": (
            "per-axis absolute maximum / (2^(bits-1) - 1), stored fp16, floored at "
            "5.960464477539063e-08 AFTER the fp16 cast; reduce axis is shape[-1] for "
            "*embed.weight and shape[0] otherwise; rank<2 tensors ship as raw fp16"
        ),
        "fp16_scale_floor": 5.960464477539063e-08,
    }


def measure_mixer_weights(rider: bytes) -> np.ndarray:
    """The 24 counted int8 SM1S mixer weights, read out of the shipped rider."""
    import importlib

    mixer = importlib.import_module("runtime.sm1_semantic_mixer")
    magic, version, count, reserved, length = mixer.HEADER.unpack_from(rider)
    if (magic, version, count, reserved) != (mixer.MAGIC, 1, 24, 0):
        raise Ren2InitError("unsupported SM1S header on the shipped rider")
    end = len(rider) - length - 24
    weights = np.frombuffer(rider[end : end + 24], dtype=np.int8)
    if weights.shape != (24,):
        raise Ren2InitError("the shipped rider does not carry 24 int8 mixer weights")
    return np.array(weights, dtype=np.int8)


def encode_member(
    state,
    *,
    representation: dict[str, Any],
    mixer_weights: np.ndarray,
    template,
    brotli_quality: int,
    brotli_lgwin: int,
    ck2: bool,
) -> dict[str, Any]:
    """The REAL member encode: pack -> SM1S -> CK2 -> brotli.  Value-dependent by design."""
    import importlib

    import brotli
    import ddm_sm3_semantic_representation as sm3

    mixer = importlib.import_module("runtime.sm1_semantic_mixer")
    body, expected, meta = sm3.pack_prune_mixed_candidate(
        state, int(representation["keep_percent"]), representation["bit_allocation"]
    )
    rider, _payload, _metadata = mixer.encode(body, template, mixer_weights)
    staged = rider
    if ck2:
        span = len(rider) & ~1
        planes = np.frombuffer(rider[:span], dtype=np.uint8)
        staged = planes[0::2].tobytes() + planes[1::2].tobytes() + rider[span:]
    member = brotli.compress(staged, quality=brotli_quality, lgwin=brotli_lgwin)
    if brotli.decompress(member) != staged:
        raise Ren2InitError("member Brotli container failed its own round trip")
    return {
        "body": body,
        "rider": rider,
        "member": member,
        "realized_state": expected,
        "pack_meta": meta,
        "body_bytes": len(body),
        "rider_bytes": len(rider),
        "member_bytes": len(member),
        "body_sha256": sha256_bytes(body),
        "rider_sha256": sha256_bytes(rider),
        "member_sha256": sha256_bytes(member),
    }


def identify_brotli_container(staged: bytes, member: bytes) -> tuple[int, int]:
    """Find the (quality, lgwin) that reproduces the shipped member BYTE-FOR-BYTE."""
    import brotli

    for quality, lgwin in BROTLI_GRID:
        if brotli.compress(staged, quality=quality, lgwin=lgwin) == member:
            return quality, lgwin
    raise Ren2InitError(
        "no searched Brotli container reproduces the shipped semantic member; the "
        "member pricer would be a look-alike rather than the shipped encoder"
    )


def build(args) -> int:
    import torch

    import ddm_jg1_seg_solve as jg1
    import ddm_up2_shipping_pose_solve as up2

    started = time.time()
    out = STORE / "init"
    out.mkdir(parents=True, exist_ok=True)
    pointer = bind_pointer()
    container = read_shipped_semantic_container()
    template = container["template"]
    body = container["body"]
    rider = container["rider"]
    member = container["member"]

    if (len(body), len(rider), len(member)) != (
        SHIPPED_BODY_BYTES,
        SHIPPED_RIDER_BYTES,
        SHIPPED_MEMBER_BYTES,
    ):
        raise Ren2InitError(
            f"shipped chain is ({len(body)}, {len(rider)}, {len(member)}) B; expected "
            f"({SHIPPED_BODY_BYTES}, {SHIPPED_RIDER_BYTES}, {SHIPPED_MEMBER_BYTES})"
        )
    if sha256_bytes(body) != SHIPPED_BODY_SHA256:
        raise Ren2InitError(f"shipped SM3R body sha drifted: {sha256_bytes(body)}")

    representation = measure_deployed_representation(body, template)
    mixer_weights = measure_mixer_weights(rider)
    quality, lgwin = identify_brotli_container(container["staged"], member)

    semantic = jg1.load_semantic_renderer(
        archive_path=POINTER_TREE / "archive.zip", runtime_dir=POINTER_TREE / "runtime"
    )
    deployed_state = {
        name: value.detach().cpu().clone() for name, value in semantic.state_dict().items()
    }

    # --- the CONTROL: re-encode the deployed state and demand byte identity ---
    control = encode_member(
        deployed_state,
        representation=representation,
        mixer_weights=mixer_weights,
        template=template,
        brotli_quality=quality,
        brotli_lgwin=lgwin,
        ck2=container["ck2_semantic"],
    )
    identity = {
        "body_identical": control["body"] == body,
        "rider_identical": control["rider"] == rider,
        "member_identical": control["member"] == member,
        "realized_state_identical": all(
            torch.equal(control["realized_state"][name], deployed_state[name])
            for name in control["realized_state"]
        ),
    }
    if not all(identity.values()):
        raise Ren2InitError(
            f"encoder control FAILED byte identity: {identity}; this producer is not "
            "the shipped encoder and must not price any member delta"
        )

    # --- the init diff against ft1's warm start ---
    ft1 = torch.load(FT1_INIT, map_location="cpu", weights_only=False)
    ft1_state = ft1["state_dict"]
    ft1_weight_identical = set(ft1_state) == set(deployed_state) and all(
        torch.equal(ft1_state[name].float(), deployed_state[name].float())
        for name in deployed_state
    )

    restored: dict[str, Any] = {
        "schema": ft1["schema"],
        "state_dict": deployed_state,
        "architecture_config": dict(ft1["architecture_config"]),
        "quant_bits": int(representation["high_bits"]),
        "deployed_representation": representation,
        "provenance": {
            "source": "move-48 archive semantic member, decoded by the shipped receiver",
            "archive_sha256": POINTER_ARCHIVE_SHA256,
            "archive_bytes": POINTER_ARCHIVE_BYTES,
            "pointer_move": 48,
            "member_sha256": sha256_bytes(member),
            "member_bytes": len(member),
            "body_sha256": sha256_bytes(body),
            "body_bytes": len(body),
            "sm1_mixer_weights_int8": [int(value) for value in mixer_weights],
            "member_container": {
                "brotli_quality": quality,
                "brotli_lgwin": lgwin,
                "ck2_plane2": container["ck2_semantic"],
                "identified_by": "byte identity against the shipped member, not by citation",
            },
            "decoded_by": "runtime/residual_archive + cpr1/ddm_mp2_semantic_receiver (shipped)",
            "restored_from": str(FT1_INIT),
            "restores": [
                "per-tensor bit-depth table (16 entries)",
                "keep_percent and the FiLM row-prune geometry",
                "the per-axis fp16 scale rule and its 5.960464477539063e-08 floor",
                "binding to the move-48 archive",
            ],
        },
    }
    init_path = out / "init_restored.pt"
    buffer = init_path.with_suffix(".pt.tmp")
    torch.save(restored, buffer)
    buffer.replace(init_path)
    init_fact = fact(init_path)

    ft1_keys = sorted(ft1)
    restored_keys = sorted(restored)
    init_diff = {
        "ft1_top_level_keys": ft1_keys,
        "restored_top_level_keys": restored_keys,
        "keys_added": sorted(set(restored_keys) - set(ft1_keys)),
        "keys_removed": sorted(set(ft1_keys) - set(restored_keys)),
        "ft1_quant_bits": ft1.get("quant_bits"),
        "restored_quant_bits": restored["quant_bits"],
        "ft1_archive_sha256": ft1["provenance"].get("archive_sha256"),
        "restored_archive_sha256": POINTER_ARCHIVE_SHA256,
        "ft1_archive_binding_is_stale": ft1["provenance"].get("archive_sha256")
        != POINTER_ARCHIVE_SHA256,
        "weight_tensors_identical_to_deployed": bool(ft1_weight_identical),
        "state_dict_tensors": len(deployed_state),
        "dropped_state_restored": [
            "bit-depth table: ft1 carried the SCALAR quant_bits=4 (a uniform grid); "
            f"the deployed table is {representation['bit_allocation']}",
            "keep_percent: ABSENT in ft1; deployed is "
            f"{representation['keep_percent']} (2 of 192 FiLM rows survive per tensor)",
            "prune mask: ABSENT in ft1; the deployed geometry is re-derivable from the "
            "descending row norm and is recorded here as measured kept rows",
            "scale rule + fp16 floor: ABSENT in ft1; recorded here",
            "archive binding: ft1 pinned cbb8d928..., three moves stale",
        ],
    }

    # measured kept rows -- the prune geometry as the deployed object actually carries it
    kept_rows: dict[str, list[int]] = {}
    for name in representation["prune_names"]:
        value = deployed_state[name]
        flat = value.reshape(value.shape[0], -1)
        kept_rows[name] = [
            int(index) for index in (flat.abs().sum(dim=1) > 0).nonzero().flatten().tolist()
        ]
    representation["measured_kept_rows"] = kept_rows

    # --- training caches ---
    tokens = jg1.load_tokens(TOKEN_FIELD)
    input_cache = out / "cache_input_coded_field.pt"
    buffer = input_cache.with_suffix(".pt.tmp")
    torch.save({"seg": torch.from_numpy(np.ascontiguousarray(tokens))}, buffer)
    buffer.replace(input_cache)

    gt_labels = jg1.load_gt_seg_labels(up2.LINEAGE_DALI)
    ft1_target = torch.load(FT1_TARGET_CACHE, map_location="cpu", weights_only=False)["seg"]
    target_matches_dali = bool((ft1_target.numpy() == gt_labels).all())
    if not target_matches_dali:
        raise Ren2InitError(
            "ft1's DALI target cache is not byte-identical to jg1.load_gt_seg_labels(DALI); "
            "it cannot be reused as the seg target"
        )

    nonzero = int(sum(int((value != 0).sum()) for value in deployed_state.values()))
    parameters = int(sum(int(value.numel()) for value in deployed_state.values()))

    result = {
        "schema": "ddm_ren2_restore_init.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": pointer,
        "pointer_move": 48,
        "rx1_header": container["rx1_header"],
        "shipped_chain": {
            "body_bytes": len(body),
            "body_sha256": sha256_bytes(body),
            "rider_bytes": len(rider),
            "rider_sha256": sha256_bytes(rider),
            "member_bytes": len(member),
            "member_sha256": sha256_bytes(member),
        },
        "deployed_representation": representation,
        "member_container": {
            "brotli_quality": quality,
            "brotli_lgwin": lgwin,
            "ck2_plane2": container["ck2_semantic"],
            "sz1_split": container["sz1_split"],
        },
        "encoder_control": {**identity, "chain": {
            "body_bytes": control["body_bytes"],
            "body_sha256": control["body_sha256"],
            "rider_bytes": control["rider_bytes"],
            "rider_sha256": control["rider_sha256"],
            "member_bytes": control["member_bytes"],
            "member_sha256": control["member_sha256"],
        }},
        "parameters": parameters,
        "nonzero_parameters": nonzero,
        "init_diff": init_diff,
        "restored_init": init_fact,
        "caches": {
            "input_coded_field": fact(input_cache),
            "target_dali_seg": fact(FT1_TARGET_CACHE),
            "target_reused_because": (
                "byte-identical to jg1.load_gt_seg_labels(LINEAGE_DALI); rebuilding it "
                "would add 118 MB to the tier for an identical payload"
            ),
            "conditioning_is_the_coded_field": True,
            "conditioning_sha256": TOKEN_FIELD_SHA256,
        },
        "producer": fact(Path(__file__)),
        "jg1": fact(Path(jg1.__file__)),
        "up2": fact(Path(up2.__file__)),
        "elapsed_seconds": time.time() - started,
    }
    retain(out / "RESULT.json", (json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
    print(json.dumps(result, indent=1, sort_keys=True), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    STORE.mkdir(parents=True, exist_ok=True)
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
