#!/usr/bin/env python3
"""Complete the scorer-free WWC1 family cone sweep on retained real objects.

The FCD1 and JF2 controls already have retained full-field screens and real
re-encodes.  This driver independently reproduces FCD1's B/H/W control, runs
the one remaining physical family (OE1) through its own online-mixture model
and RC64 receiver trajectory, and prices the cross-family coordinate union on
the JF2 k060 receiving body with one real joint re-encode plus a deterministic
repeat.

No scorer or remote service is loaded.  Every materialized field, archive,
stream, checkpoint, and receipt is retained below the chartered Vertigo root.
Byte-identical OE1 fields share one content-addressed screen/decoded payload;
the per-rung receipts prove the identity instead of storing duplicate bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_bhw2_jf2_oe1_argmax_screen as bhw2
from experiments import ddm_fcd1_field_for_coder_diagonal as fcd1
from experiments import ddm_jg2_tail_reencode as jg2
from experiments import ddm_oe1_online_escape_member as oe1

VERTIGO_ROOT = Path("/Volumes/VertigoDataTier/pact")
AP_ROOT = Path("/Volumes/APDataStore/pact")
STORE = VERTIGO_ROOT / "ddm_wwc1_winwin_cone_sweep"
CHARTER = REPO / ".omx/research/charters/ddm_wwc1_winwin_cone_sweep_20260831.md"
COMMON = REPO / ".omx/tmp/codex_runs/_common_contract.md"
FCD1_MEMO = REPO / ".omx/research/ddm_fcd1_field_for_coder_diagonal_20260829.md"
FCD2_MEMO = REPO / ".omx/research/ddm_fcd2_distortion_legs_execute_20260829.md"
FCD3_MEMO = REPO / ".omx/research/ddm_fcd3_pose_screened_reselection_20260829.md"
BHW1_MEMO = REPO / ".omx/research/ddm_bhw1_winwin_cone_rescreen_20260829.md"
BHW2_MEMO = REPO / ".omx/research/ddm_bhw2_jf2_oe1_argmax_screen_20260829.md"
GESTALT_MEMO = REPO / ".omx/research/ddm_gest1_decoupling_gestalt_20260831.md"
FRONTIER = REPO / ".omx/state/canonical_frontier_pointer.json"

FCD1_RESULT = AP_ROOT / "ddm_fcd1_field_for_coder_diagonal/BYTE_ONLY_RESULT.json"
FCD1_COORDS = (
    AP_ROOT / "ddm_fcd1_field_for_coder_diagonal/retained/coordinates/benefit_pool.frame_y_x_old_new_assignment.npz"
)
BHW2_RESULT = AP_ROOT / "ddm_bhw2_jf2_oe1_argmax_screen/jf2/JF2_RESULT.json"
BHW2_MANIFEST = AP_ROOT / "ddm_bhw2_jf2_oe1_argmax_screen/MANIFEST.json"
BHW2_JF2_COORDS = AP_ROOT / "ddm_bhw2_jf2_oe1_argmax_screen/jf2/screen/k060000/retained/benefit.frame_y_x_old_new.npz"
BHW2_JF2_ARGMAX = (
    AP_ROOT / "ddm_bhw2_jf2_oe1_argmax_screen/jf2/base_argmax_replay/retained/argmax/jf2_k060000.coding_argmax.u8.bin"
)

JF2_ROOT = AP_ROOT / "ddm_jf2_terminal_diagonal_harvest/retained/k060000"
JF2_ARCHIVE = JF2_ROOT / "retained/candidate_archive.zip"
JF2_TOKENS = JF2_ROOT / "retained/decoded_tokens.u8"
OE1_SOURCE_ROOT = REPO / ".omx/tmp/arm_receipts_local/ddm_oe1_online_escape_member"

N, HEIGHT, WIDTH = 600, 384, 512
PLANE = HEIGHT * WIDTH
POSITIONS = N * PLANE
STAGE_FRAMES = 20
AXIS = "[macOS-CPU advisory / scorer-free exact B/H/W and real RC64 re-encode]"
S_PER_BYTE = 25.0 / 37_545_489.0
CONTROL_PROJECTED_BYTES = 128 << 20
OE1_BASE_PROJECTED_BYTES = 1_400_000_000
OE1_UNIQUE_SCREEN_PROJECTED_BYTES = 520_000_000
OE1_TAIL_PROJECTED_BYTES = 320_000_000
COMPOSE_PROJECTED_BYTES = 512 << 20
RESERVE_BYTES = 1 << 30
CONTROL_SCHEMA = "ddm_wwc1_control_stage.v1"
SHARED_DECODE_SCHEMA = "ddm_wwc1_oe1_shared_decode_stage.v1"
CONTROL_ROOT = STORE / "control_v2"
CONTROL_RESULT_PATH = CONTROL_ROOT / "CONTROL_RESULT.json"


class Wwc1Error(RuntimeError):
    """A custody, storage, identity, resume, or real-coder gate refused."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_fact(fact: Mapping[str, Any]) -> dict[str, Any]:
    observed = file_fact(Path(str(fact["path"])))
    if observed != dict(fact):
        raise Wwc1Error(f"artifact drifted: expected={dict(fact)}, observed={observed}")
    return observed


def verify_embedded_facts(value: object) -> int:
    """Rehash every file-fact nested in a completed stage result."""
    verified = 0
    if isinstance(value, Mapping):
        if {"path", "bytes", "sha256"}.issubset(value):
            expected = {key: value[key] for key in ("path", "bytes", "sha256")}
            verify_fact(expected)
            verified += 1
        for child in value.values():
            verified += verify_embedded_facts(child)
    elif isinstance(value, list | tuple):
        for child in value:
            verified += verify_embedded_facts(child)
    return verified


def require_fact(path: Path, *, size: int | None = None, digest: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise Wwc1Error(f"required source is absent: {path}")
    observed = file_fact(path)
    if size is not None and observed["bytes"] != size:
        raise Wwc1Error(f"required source size drifted: {observed}")
    if digest is not None and observed["sha256"] != digest:
        raise Wwc1Error(f"required source hash drifted: {observed}")
    return observed


def atomic_json(path: Path, payload: object) -> dict[str, Any]:
    return bhw2.atomic_json(path, payload)


def atomic_npz(path: Path, **values: np.ndarray) -> dict[str, Any]:
    return bhw2.atomic_npz(path, **values)


def storage_preflight(phase: str, projected_bytes: int) -> dict[str, Any]:
    STORE.mkdir(parents=True, exist_ok=True)
    if not STORE.resolve().is_relative_to(VERTIGO_ROOT.resolve()):
        raise Wwc1Error(f"WWC1 store escaped Vertigo: {STORE}")
    free = shutil.disk_usage(STORE).free
    required = projected_bytes + RESERVE_BYTES
    result = {
        "schema": "ddm_wwc1_storage_preflight.v1",
        "phase": phase,
        "selected_tier": "VertigoDataTier",
        "path": str(STORE),
        "free_bytes_before": free,
        "projected_materialization_bytes": projected_bytes,
        "post_run_reserve_bytes": RESERVE_BYTES,
        "required_free_bytes": required,
        "status": "PASS" if free >= required else "BLOCKED_SHORTFALL",
        "shortfall_bytes": max(0, required - free),
        "cleanup_attempted": False,
        "ap_fallback_free_bytes_observed": shutil.disk_usage(AP_ROOT).free,
    }
    atomic_json(STORE / "preflight" / f"{phase}.json", result)
    if free < required:
        raise Wwc1Error(
            f"storage preflight blocked {phase}: free={free}, required={required}, "
            f"shortfall={required - free}; no deletion or local-disk routing attempted"
        )
    return result


def oe1_source_streams() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for window in oe1.WINDOWS:
        name = oe1.label(window)
        path = OE1_SOURCE_ROOT / "retained/rungs" / name / "tokens.rc64"
        expected_bytes, expected_sha = bhw2.OE1_SOURCE_PINS[name]["stream"]
        rows[name] = require_fact(path, size=expected_bytes, digest=expected_sha)
    return rows


def source_binding() -> dict[str, Any]:
    bhw2_result = json.loads(BHW2_RESULT.read_text())
    expected_jf2_coords = bhw2_result["screen"]["payloads"]["benefit_coordinates"]
    observed_jf2_coords = file_fact(BHW2_JF2_COORDS)
    if observed_jf2_coords != expected_jf2_coords:
        raise Wwc1Error(
            "BHW2 JF2 coordinate payload drifted from its landed result: "
            f"expected={expected_jf2_coords}, observed={observed_jf2_coords}"
        )
    return {
        "schema": "ddm_wwc1_source_binding.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "implementation": file_fact(Path(__file__)),
        "governance": {
            "charter": file_fact(CHARTER),
            "common_contract": file_fact(COMMON),
            "fcd1": file_fact(FCD1_MEMO),
            "fcd2": file_fact(FCD2_MEMO),
            "fcd3": file_fact(FCD3_MEMO),
            "bhw1": file_fact(BHW1_MEMO),
            "bhw2": file_fact(BHW2_MEMO),
            "gestalt": file_fact(GESTALT_MEMO),
            "frontier_pointer": file_fact(FRONTIER),
        },
        "control_inputs": {
            "tokens": require_fact(
                fcd1.TOKENS,
                size=fcd1.PINS["tokens"][0],
                digest=fcd1.PINS["tokens"][1],
            ),
            "coding_argmax": require_fact(
                fcd1.CODING_ARGMAX,
                size=fcd1.PINS["coding_argmax"][0],
                digest=fcd1.PINS["coding_argmax"][1],
            ),
            "dali_gt": require_fact(
                fcd1.GT_ARGMAX,
                size=fcd1.PINS["gt_argmax"][0],
                digest=fcd1.PINS["gt_argmax"][1],
            ),
            "fcd1_result": file_fact(FCD1_RESULT),
            "fcd1_coordinates": require_fact(
                FCD1_COORDS,
                size=27_105,
                digest="cc09fd9d4cb9a7253df30dbe38d5f60e33ee9e62c8217d9d0b1276ea5c2b5042",
            ),
        },
        "jf2": {
            "archive": require_fact(
                JF2_ARCHIVE,
                size=178_792,
                digest="59428f07e6344129d2c5e37ffac84ec19f8e609b2b5951d0d970fb694b88c54a",
            ),
            "tokens": require_fact(
                JF2_TOKENS,
                size=POSITIONS,
                digest="15018481bd8007dd9099d1b67d5e8014283465d062a34ba3f06b3450758b5878",
            ),
            "coding_argmax": require_fact(
                BHW2_JF2_ARGMAX,
                size=POSITIONS,
                digest="d3dfc1c8b3816fcd609d4b64682e8d9a1c73825d5a91ba703af2255502e424ab",
            ),
            "benefit_coordinates": observed_jf2_coords,
            "result": file_fact(BHW2_RESULT),
            "manifest": file_fact(BHW2_MANIFEST),
        },
        "oe1": {
            "tokens": require_fact(
                oe1.TOKENS,
                size=POSITIONS,
                digest="cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb",
            ),
            "source_streams": oe1_source_streams(),
            "result": file_fact(OE1_SOURCE_ROOT / "RESULT.json"),
            "manifest": file_fact(OE1_SOURCE_ROOT / "MANIFEST.json"),
        },
    }


def _control_stage_root(start: int, end: int) -> Path:
    return CONTROL_ROOT / "stages" / f"frames_{start:04d}_{end - 1:04d}"


def run_control() -> dict[str, Any]:
    result_path = CONTROL_RESULT_PATH
    if result_path.is_file():
        prior = json.loads(result_path.read_text())
        current_binding_sha = sha256_json(source_binding())
        if prior.get("source_binding_sha256") != current_binding_sha:
            raise Wwc1Error("completed FCD1 control source binding drifted")
        verify_embedded_facts(prior)
        for receipt_fact in prior["stage_receipts"]:
            receipt = json.loads(Path(receipt_fact["path"]).read_text())
            verify_embedded_facts(receipt)
        return prior
    preflight = storage_preflight("control", CONTROL_PROJECTED_BYTES)
    binding = source_binding()
    binding_sha = sha256_json(binding)
    tokens = np.memmap(fcd1.TOKENS, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    argmax = np.memmap(fcd1.CODING_ARGMAX, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    gt = np.load(fcd1.GT_ARGMAX, mmap_mode="r", allow_pickle=False)
    with np.load(FCD1_COORDS, allow_pickle=False) as payload:
        union_coords = np.asarray(payload["coords"], dtype=np.int32).copy()
        union_old = np.asarray(payload["old"], dtype=np.uint8).copy()
        union_new = np.asarray(payload["new"], dtype=np.uint8).copy()
    raw_totals = Counter()
    raw_independent_totals = Counter()
    union_totals = Counter()
    union_independent_totals = Counter()
    receipt_facts: list[dict[str, Any]] = []
    started = time.perf_counter()
    for start in range(0, N, STAGE_FRAMES):
        end = min(start + STAGE_FRAMES, N)
        root = _control_stage_root(start, end)
        receipt_path = root / "RECEIPT.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("schema") != CONTROL_SCHEMA or receipt.get("source_binding_sha256") != binding_sha:
                raise Wwc1Error(f"control stage binding drifted: {receipt_path}")
            for fact in receipt["artifacts"].values():
                verify_fact(fact)
            for key in ("B", "H", "W"):
                raw_totals[key] += int(receipt["raw_counts"][key])
                raw_independent_totals[key] += int(receipt["raw_independent_counts"][key])
                union_totals[key] += int(receipt["union_counts"][key])
                union_independent_totals[key] += int(receipt["union_independent_counts"][key])
            receipt_facts.append(file_fact(receipt_path))
            continue
        token_block = np.asarray(tokens[start:end])
        argmax_block = np.asarray(argmax[start:end])
        gt_block = np.asarray(gt[start:end])
        raw_canonical = fcd1.classify_pool(token_block, argmax_block, gt_block)
        raw_changed = token_block != argmax_block
        raw_independent = {
            "B": raw_changed & (token_block != gt_block) & (argmax_block == gt_block),
            "H": raw_changed & (token_block == gt_block) & (argmax_block != gt_block),
            "W": raw_changed & (token_block != gt_block) & (argmax_block != gt_block),
        }
        raw_canonical_by_name = {
            "B": raw_canonical["benefit"],
            "H": raw_canonical["harm"],
            "W": raw_canonical["wash"],
        }
        if any(not np.array_equal(raw_canonical_by_name[key], raw_independent[key]) for key in raw_independent):
            raise Wwc1Error(f"independent raw-pool B/H/W implementation disagreed at frames {start}:{end}")

        union_block = token_block.copy()
        selected = (union_coords[:, 0] >= start) & (union_coords[:, 0] < end)
        local_coords = union_coords[selected].copy()
        local_coords[:, 0] -= start
        local_index = tuple(np.moveaxis(local_coords, 1, 0))
        if not np.array_equal(token_block[local_index], union_old[selected]):
            raise Wwc1Error(f"banked-union old labels drifted at frames {start}:{end}")
        if not np.array_equal(argmax_block[local_index], union_new[selected]):
            raise Wwc1Error(f"banked-union coding-argmax labels drifted at frames {start}:{end}")
        union_block[local_index] = union_new[selected]
        union_canonical = fcd1.classify_pool(token_block, union_block, gt_block)
        union_changed = token_block != union_block
        union_independent = {
            "B": union_changed & (token_block != gt_block) & (union_block == gt_block),
            "H": union_changed & (token_block == gt_block) & (union_block != gt_block),
            "W": union_changed & (token_block != gt_block) & (union_block != gt_block),
        }
        union_canonical_by_name = {
            "B": union_canonical["benefit"],
            "H": union_canonical["harm"],
            "W": union_canonical["wash"],
        }
        if any(not np.array_equal(union_canonical_by_name[key], union_independent[key]) for key in union_independent):
            raise Wwc1Error(f"independent banked-union B/H/W implementation disagreed at frames {start}:{end}")
        artifacts: dict[str, dict[str, Any]] = {}
        raw_counts: dict[str, int] = {}
        raw_independent_counts: dict[str, int] = {}
        union_counts: dict[str, int] = {}
        union_independent_counts: dict[str, int] = {}
        for scope, canonical, independent in (
            ("raw", raw_canonical_by_name, raw_independent),
            ("union", union_canonical_by_name, union_independent),
        ):
            for key, mask in independent.items():
                count = int(canonical[key].sum())
                independent_count = int(mask.sum())
                if scope == "raw":
                    raw_counts[key] = count
                    raw_independent_counts[key] = independent_count
                else:
                    union_counts[key] = count
                    union_independent_counts[key] = independent_count
                packed = np.packbits(mask.reshape(mask.shape[0], -1), axis=1, bitorder="little")
                artifacts[f"{scope}_{key}_packed"] = atomic_npz(
                    root / f"{scope}_{key}.mask.packbits.npz",
                    packed=packed,
                    frame_start=np.asarray([start], dtype=np.int32),
                    frame_end=np.asarray([end], dtype=np.int32),
                    shape=np.asarray(mask.shape, dtype=np.int32),
                )
        local_b = np.argwhere(union_independent["B"]).astype(np.int32)
        if local_b.size:
            local_b[:, 0] += start
            union_b_index = tuple(np.moveaxis(np.argwhere(union_independent["B"]), 1, 0))
            old = token_block[union_b_index].astype(np.uint8)
            new = union_block[union_b_index].astype(np.uint8)
        else:
            old = np.empty(0, dtype=np.uint8)
            new = np.empty(0, dtype=np.uint8)
        artifacts["B_coordinates"] = atomic_npz(
            root / "B.frame_y_x_old_new.npz",
            coords=local_b,
            old=old,
            new=new,
        )
        receipt = {
            "schema": CONTROL_SCHEMA,
            "source_binding_sha256": binding_sha,
            "frame_start": start,
            "frame_end": end,
            "raw_counts": raw_counts,
            "raw_independent_counts": raw_independent_counts,
            "union_counts": union_counts,
            "union_independent_counts": union_independent_counts,
            "artifacts": artifacts,
        }
        atomic_json(receipt_path, receipt)
        receipt_facts.append(file_fact(receipt_path))
        raw_totals.update(raw_counts)
        raw_independent_totals.update(raw_independent_counts)
        union_totals.update(union_counts)
        union_independent_totals.update(union_independent_counts)
    raw_observed = {key: int(raw_totals[key]) for key in ("B", "H", "W")}
    raw_independent_observed = {key: int(raw_independent_totals[key]) for key in ("B", "H", "W")}
    union_observed = {key: int(union_totals[key]) for key in ("B", "H", "W")}
    union_independent_observed = {key: int(union_independent_totals[key]) for key in ("B", "H", "W")}
    raw_expected = {"B": 5_268, "H": 221_862, "W": 541}
    union_expected = {"B": 5_268, "H": 0, "W": 0}
    if raw_observed != raw_expected or raw_independent_observed != raw_expected:
        raise Wwc1Error(
            "FCD1 raw-pool positive control failed: "
            f"canonical={raw_observed}, independent={raw_independent_observed}, expected={raw_expected}"
        )
    if union_observed != union_expected or union_independent_observed != union_expected:
        raise Wwc1Error(
            "FCD1 banked-union positive control failed: "
            f"canonical={union_observed}, independent={union_independent_observed}, expected={union_expected}"
        )
    result = {
        "schema": "ddm_wwc1_control_result.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "preflight": preflight,
        "source_binding_sha256": binding_sha,
        "population_positions": POSITIONS,
        "counts": union_observed,
        "independent_repeat_counts": union_independent_observed,
        "disagreement_population": sum(union_observed.values()),
        "raw_model_disagreement_counts": raw_observed,
        "raw_model_disagreement_independent_counts": raw_independent_observed,
        "raw_model_disagreement_population": sum(raw_observed.values()),
        "control_reproduced": True,
        "dali_gt_used": binding["control_inputs"]["dali_gt"],
        "stage_receipts": receipt_facts,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(result_path, result)
    return result


def _shared_decode_stage_root(root: Path, start: int, end: int) -> Path:
    return root / "stages" / f"frames_{start:04d}_{end - 1:04d}"


def verify_oe1_shared_decode(
    *,
    root: Path,
    target_path: Path,
    streams: Mapping[str, Mapping[str, Any]],
    windows: Iterable[int],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    selected_windows = tuple(int(value) for value in windows)
    names = tuple(oe1.label(value) for value in selected_windows)
    target_fact = file_fact(target_path)
    stream_facts = {name: verify_fact(streams[name]) for name in names}
    decode_binding = {
        "source_binding_sha256": sha256_json(binding),
        "target": target_fact,
        "streams": stream_facts,
        "windows": selected_windows,
        "payload_custody_mode": "content-addressed shared target; every decoder must reproduce these exact bytes",
    }
    binding_sha = sha256_json(decode_binding)
    result_path = root / "DECODE_RESULT.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text())
        if prior.get("decode_binding_sha256") != binding_sha:
            raise Wwc1Error(f"completed shared decode binding drifted: {result_path}")
        verify_fact(prior["shared_decoded_payload"])
        return prior

    bhw2.purge_runtime_modules()
    library, rc64 = oe1.compile_rc64(root)
    runtime = oe1.load_receiver(library)
    states = {window: oe1.EscapeState(window) for window in selected_windows}
    decoders = {
        window: rc64.NativeDecoder(library, Path(stream_facts[oe1.label(window)]["path"]).read_bytes())
        for window in selected_windows
    }
    receipts = bhw2.contiguous_receipts(root, schema=SHARED_DECODE_SCHEMA, binding_sha256=binding_sha)
    if receipts:
        last = receipts[-1]
        last_root = _shared_decode_stage_root(root, int(last["frame_start"]), int(last["frame_end"]))
        start_frame, previous = oe1.restore_common_state(
            runtime,
            states,
            last_root / "receiver_state.npz",
            SHARED_DECODE_SCHEMA,
        )
        with np.load(last_root / "decoder_states.npz", allow_pickle=False) as saved:
            for window in selected_windows:
                oe1.bl1.restore_decoder_state(decoders[window], saved[oe1.label(window)])
    else:
        start_frame = 0
        torch = runtime["torch"]
        previous = torch.zeros((1, HEIGHT, WIDTH), dtype=torch.long, device=runtime["device"])

    target = np.memmap(target_path, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    torch = runtime["torch"]
    model = runtime["model"]
    sparse = runtime["sparse"]
    corrector = runtime["corrector"]
    residual = runtime["residual"]
    parts = runtime["parts"]
    device = runtime["device"]
    started = time.perf_counter()
    with torch.inference_mode():
        for stage_start in range(start_frame, N, STAGE_FRAMES):
            stage_end = min(stage_start + STAGE_FRAMES, N)
            stage_digest = hashlib.sha256()
            for frame in range(stage_start, stage_end):
                current = torch.zeros_like(previous)
                index = torch.tensor([frame], dtype=torch.long, device=device)
                context = model.prepare_frame_context(index, previous)
                if frame:
                    previous_cpu = previous[0].to(device="cpu", dtype=torch.uint8).numpy()
                    boundary = residual._boundary_buckets(previous_cpu).reshape(-1)
                else:
                    boundary = np.full(PLANE, 4, dtype=np.uint8)
                corrector.begin_frame(boundary)
                expected_plane = np.asarray(target[frame]).reshape(-1)
                stage_digest.update(expected_plane.tobytes())
                for group, (device_positions, flat_positions) in enumerate(runtime["plans"]):
                    base_logits = sparse.selected_logits(current, context, group).cpu().numpy()
                    predicted = base_logits.argmax(axis=1).astype(np.int64)
                    feature = boundary[flat_positions].astype(np.int64) * bhw2.CLASSES + predicted
                    corrected = base_logits + parts.table.values[feature]
                    probability = residual._probability_table(corrected, runtime["renderer"].HPAC_LOGIT_PRECISION)
                    receiver_state = corrector.group_state(probability, predicted, flat_positions)
                    base = np.asarray(corrector.coding_row(receiver_state), dtype=np.float32)
                    expected = expected_plane[flat_positions].astype(np.int64)
                    base_frequency, _costs = oe1.selected_costs(rc64, base, expected)
                    selected = base_frequency[np.arange(expected.size), expected]
                    anti = selected.astype(np.uint64) * bhw2.CLASSES < oe1.TOTAL
                    for window in selected_windows:
                        candidate = states[window].coding(base, group, feature)
                        decoded = decoders[window].decode(candidate).astype(np.int64)
                        if not np.array_equal(decoded, expected):
                            mismatch = int(np.flatnonzero(decoded != expected)[0])
                            raise Wwc1Error(
                                "OE1 candidate receiver mismatch: "
                                f"rung={oe1.label(window)}, frame={frame}, group={group}, within_group={mismatch}"
                            )
                        states[window].observe(frame, group, feature, anti)
                    corrector.observe(receiver_state, expected)
                    current.reshape(-1)[device_positions] = torch.from_numpy(expected).to(device)
                frame_tokens = current[0].to(device="cpu", dtype=torch.uint8).numpy()
                if not np.array_equal(frame_tokens.reshape(-1), expected_plane):
                    raise Wwc1Error(f"OE1 shared receiver trajectory drifted at frame {frame}")
                corrector.end_frame(frame_tokens.reshape(-1))
                previous = current

            current_root = _shared_decode_stage_root(root, stage_start, stage_end)
            decoder_arrays = {oe1.label(window): oe1.bl1.decoder_state(decoders[window]) for window in selected_windows}
            artifacts = {
                "decoder_states": atomic_npz(current_root / "decoder_states.npz", **decoder_arrays),
                "receiver_state": atomic_npz(
                    current_root / "receiver_state.npz",
                    **oe1.common_state_arrays(
                        runtime,
                        states,
                        SHARED_DECODE_SCHEMA,
                        stage_end,
                        previous[0].to(device="cpu", dtype=torch.uint8).numpy(),
                    ),
                ),
            }
            receipt = {
                "schema": SHARED_DECODE_SCHEMA,
                "execution_binding_sha256": binding_sha,
                "frame_start": stage_start,
                "frame_end": stage_end,
                "decoded_payload_identity": {name: True for name in names},
                "decoded_stage_sha256": stage_digest.hexdigest(),
                "shared_payload": target_fact,
                "artifacts": artifacts,
                "elapsed_seconds": time.perf_counter() - started,
            }
            atomic_json(current_root / "RECEIPT.json", receipt)
            print(
                json.dumps(
                    {
                        "family": "oe1",
                        "phase": "shared_candidate_decode",
                        "frame_end": stage_end,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    rows = {
        oe1.label(window): {
            "decoded_tokens": target_fact,
            "decoded_field_identity": True,
            "decoder_bit_position": int(decoders[window].bit_position),
            "payload_storage": "shared_content_addressed_exact_bytes",
        }
        for window in selected_windows
    }
    result = {
        "schema": "ddm_wwc1_oe1_shared_decode_result.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "decode_binding_sha256": binding_sha,
        "shared_decoded_payload": target_fact,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(result_path, result)
    return result


def _oe1_base_archives() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for window in oe1.WINDOWS:
        name = oe1.label(window)
        member_path = OE1_SOURCE_ROOT / "retained/rungs" / name / "member.bin"
        expected_bytes, expected_sha = bhw2.OE1_SOURCE_PINS[name]["member"]
        require_fact(member_path, size=expected_bytes, digest=expected_sha)
        destination = STORE / "oe1/retained/base_archives" / f"{name}.archive.zip"
        jg2.pack_archive(member_path.read_bytes(), destination)
        output[name] = file_fact(destination)
    return output


def run_oe1() -> dict[str, Any]:
    result_path = STORE / "oe1/OE1_RESULT.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text())
        current_binding_sha = sha256_json(source_binding())
        if prior.get("source_binding_sha256") != current_binding_sha:
            raise Wwc1Error("completed OE1 source binding drifted")
        verify_embedded_facts(prior)
        return prior
    control = run_control()
    if not control.get("control_reproduced"):
        raise Wwc1Error("OE1 refuses before the FCD1 control reproduces")
    preflight = storage_preflight("oe1_base", OE1_BASE_PROJECTED_BYTES)
    binding = source_binding()
    atomic_json(STORE / "SOURCE_BINDING.json", binding)
    source_streams = oe1_source_streams()
    base_replay = bhw2.run_replay(
        family="oe1",
        root=STORE / "oe1/base_argmax_replay",
        target_path=oe1.TOKENS,
        binding=binding,
        retain_argmax=True,
        windows=oe1.WINDOWS,
        expected_streams=source_streams,
    )
    hash_groups: dict[str, list[str]] = {}
    for name, fact in base_replay["coding_argmax"].items():
        hash_groups.setdefault(str(fact["sha256"]), []).append(name)
    unique_count = len(hash_groups)
    remainder_projection = unique_count * OE1_UNIQUE_SCREEN_PROJECTED_BYTES + OE1_TAIL_PROJECTED_BYTES
    remainder_preflight = storage_preflight("oe1_after_argmax", remainder_projection)

    screens_by_hash: dict[str, dict[str, Any]] = {}
    screens_by_name: dict[str, dict[str, Any]] = {}
    for digest, names in sorted(hash_groups.items()):
        representative = names[0]
        screen = bhw2.classify_and_materialize(
            row=f"oe1_argmax_{digest[:12]}",
            root=STORE / "oe1/screen_unique" / digest,
            tokens_path=oe1.TOKENS,
            argmax_path=Path(base_replay["coding_argmax"][representative]["path"]),
            binding=binding,
        )
        screens_by_hash[digest] = screen
        for name in names:
            screens_by_name[name] = screen

    benefit_groups: dict[str, list[int]] = {}
    for window in oe1.WINDOWS:
        name = oe1.label(window)
        benefit_sha = str(screens_by_name[name]["payloads"]["benefit_field"]["sha256"])
        benefit_groups.setdefault(benefit_sha, []).append(window)
    candidate_streams: dict[str, dict[str, Any]] = {}
    repeat_streams: dict[str, dict[str, Any]] = {}
    replays: list[dict[str, Any]] = []
    repeat_replays: list[dict[str, Any]] = []
    decode_rows: dict[str, dict[str, Any]] = {}
    decode_receipts: list[dict[str, Any]] = []
    for group_index, windows in enumerate(benefit_groups.values()):
        first = oe1.label(windows[0])
        target_path = Path(screens_by_name[first]["payloads"]["benefit_field"]["path"])
        replay = bhw2.run_replay(
            family="oe1",
            root=STORE / "oe1/benefit_reencode" / f"group_{group_index:02d}",
            target_path=target_path,
            binding=binding,
            retain_argmax=False,
            windows=windows,
        )
        repeat = bhw2.run_replay(
            family="oe1",
            root=STORE / "oe1/benefit_reencode_repeat" / f"group_{group_index:02d}",
            target_path=target_path,
            binding=binding,
            retain_argmax=False,
            windows=windows,
        )
        for window in windows:
            name = oe1.label(window)
            replay_identity = {key: replay["streams"][name][key] for key in ("bytes", "sha256")}
            repeat_identity = {key: repeat["streams"][name][key] for key in ("bytes", "sha256")}
            if replay_identity != repeat_identity:
                raise Wwc1Error(f"OE1 deterministic re-encode repeat drifted for {name}")
        replays.append(replay)
        repeat_replays.append(repeat)
        candidate_streams.update(replay["streams"])
        repeat_streams.update(repeat["streams"])
        group_streams = {oe1.label(window): replay["streams"][oe1.label(window)] for window in windows}
        decode = verify_oe1_shared_decode(
            root=STORE / "oe1/benefit_decode" / f"group_{group_index:02d}",
            target_path=target_path,
            streams=group_streams,
            windows=windows,
            binding=binding,
        )
        decode_receipts.append(decode)
        decode_rows.update(decode["rows"])

    base_archives = _oe1_base_archives()
    prefix = oe1.member_prefix(STORE / "oe1/member_prefix")[0]
    rows: list[dict[str, Any]] = []
    for window in oe1.WINDOWS:
        name = oe1.label(window)
        member_path = STORE / "oe1/retained/benefit_members" / f"{name}.member.bin"
        bhw2.atomic_bytes(member_path, prefix + Path(candidate_streams[name]["path"]).read_bytes())
        candidate_path = STORE / "oe1/retained/benefit_archives" / f"{name}.archive.zip"
        jg2.pack_archive(member_path.read_bytes(), candidate_path)
        repeat_member_path = STORE / "oe1/retained/benefit_repeat_members" / f"{name}.member.bin"
        bhw2.atomic_bytes(
            repeat_member_path,
            prefix + Path(repeat_streams[name]["path"]).read_bytes(),
        )
        repeat_candidate_path = STORE / "oe1/retained/benefit_repeat_archives" / f"{name}.archive.zip"
        jg2.pack_archive(repeat_member_path.read_bytes(), repeat_candidate_path)
        repeat_archive = file_fact(repeat_candidate_path)
        parseback = bhw2.verify_oe1_archive_stream(
            archive=candidate_path,
            member=member_path,
            stream=Path(candidate_streams[name]["path"]),
        )
        row = bhw2.byte_row(
            row=f"oe1_{name}",
            screen=screens_by_name[name],
            base_archive=base_archives[name],
            candidate_archive=file_fact(candidate_path),
        )
        row["coding_argmax"] = base_replay["coding_argmax"][name]
        row["shared_argmax_group_sha256"] = row["coding_argmax"]["sha256"]
        row["receiver_decode"] = decode_rows[name]
        row["archive_parseback"] = parseback
        row["deterministic_repeat_stream"] = repeat_streams[name]
        row["deterministic_repeat_archive"] = repeat_archive
        row["deterministic_repeat_identical"] = (
            repeat_archive["bytes"] == row["candidate_archive"]["bytes"]
            and repeat_archive["sha256"] == row["candidate_archive"]["sha256"]
        )
        if not row["deterministic_repeat_identical"]:
            raise Wwc1Error(f"OE1 deterministic repeat archive drifted for {name}")
        rows.append(row)
    result = {
        "schema": "ddm_wwc1_oe1_result.v1",
        "complete": True,
        "terminal": True,
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "preflight": preflight,
        "remainder_preflight": remainder_preflight,
        "source_binding_sha256": sha256_json(binding),
        "control": file_fact(CONTROL_RESULT_PATH),
        "base_replay": base_replay,
        "unique_argmax_count": unique_count,
        "argmax_hash_groups": hash_groups,
        "screens_by_hash": screens_by_hash,
        "benefit_reencodes": replays,
        "benefit_reencode_repeats": repeat_replays,
        "benefit_decodes": decode_receipts,
        "rows": rows,
        "family_verdict": (
            "BYTE-ADMITTED-CONE" if any(row["verdict"] == "BYTE-ADMITTED-FIRE-MAIN" for row in rows) else "BYTE-REFUSED"
        ),
        "scorer_ran_here": False,
        "seal_created_here": False,
    }
    atomic_json(result_path, result)
    bhw2.write_jsonl(STORE / "oe1/OE1_ROWS.jsonl", rows)
    return result


def _load_coords(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        coords = np.asarray(payload["coords"], dtype=np.int32).copy()
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise Wwc1Error(f"coordinate payload has wrong shape: {path}: {coords.shape}")
    return coords


def _materialize_joint_union(oe1_result: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    fcd_coords = _load_coords(FCD1_COORDS)
    jf2_coords = _load_coords(BHW2_JF2_COORDS)
    oe1_coord_parts = [
        _load_coords(Path(screen["payloads"]["benefit_coordinates"]["path"]))
        for screen in oe1_result["screens_by_hash"].values()
    ]
    oe1_coords = np.concatenate(oe1_coord_parts, axis=0)
    sources = ((1, fcd_coords), (2, jf2_coords), (4, oe1_coords))
    membership: dict[int, int] = {}
    for bit, coords in sources:
        linear = coords[:, 0].astype(np.int64) * PLANE + coords[:, 1].astype(np.int64) * WIDTH + coords[:, 2]
        for index in linear:
            membership[int(index)] = membership.get(int(index), 0) | bit
    linear_union = np.asarray(sorted(membership), dtype=np.int64)
    frames = linear_union // PLANE
    within = linear_union % PLANE
    yy = within // WIDTH
    xx = within % WIDTH
    coords = np.stack([frames, yy, xx], axis=1).astype(np.int32)
    membership_bits = np.asarray([membership[int(index)] for index in linear_union], dtype=np.uint8)
    base = np.memmap(JF2_TOKENS, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    gt = np.load(fcd1.GT_ARGMAX, mmap_mode="r", allow_pickle=False)
    old = np.asarray(base[frames, yy, xx], dtype=np.uint8)
    new = np.asarray(gt[frames, yy, xx], dtype=np.uint8)
    changed = old != new
    coord_fact = atomic_npz(
        STORE / "compose/retained/joint_union.frame_y_x_old_new_membership.npz",
        coords=coords,
        old=old,
        new=new,
        membership_bits=membership_bits,
        changes_base_jf2=changed,
    )
    destination = STORE / "compose/retained/joint_union_on_jf2_k060.tokens.u8"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".partial.{os.getpid()}")
    by_frame: dict[int, np.ndarray] = {}
    for frame in np.unique(frames[changed]):
        by_frame[int(frame)] = np.flatnonzero(changed & (frames == frame))
    try:
        with temporary.open("wb") as handle:
            for frame in range(N):
                plane = np.asarray(base[frame], dtype=np.uint8).copy()
                selected = by_frame.get(frame)
                if selected is not None:
                    plane[yy[selected], xx[selected]] = new[selected]
                handle.write(plane.tobytes(order="C"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    field_fact = file_fact(destination)
    if field_fact["bytes"] != POSITIONS:
        raise Wwc1Error(f"joint union field has wrong size: {field_fact}")
    metadata = {
        "schema": "ddm_wwc1_joint_union_field.v1",
        "receiving_body": "JF2 k060",
        "source_coordinate_counts": {
            "fcd1": int(fcd_coords.shape[0]),
            "jf2_k060": int(jf2_coords.shape[0]),
            "oe1": int(oe1_coords.shape[0]),
        },
        "unique_coordinate_count": int(coords.shape[0]),
        "actual_edits_on_jf2_body": int(changed.sum()),
        "membership_histogram": {
            str(value): int((membership_bits == value).sum()) for value in np.unique(membership_bits)
        },
        "coordinates": coord_fact,
        "field": field_fact,
    }
    atomic_json(STORE / "compose/JOINT_FIELD.json", metadata)
    return field_fact, metadata


def run_compose() -> dict[str, Any]:
    result_path = STORE / "compose/COMPOSE_RESULT.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text())
        current_binding_sha = sha256_json(source_binding())
        if prior.get("source_binding_sha256") != current_binding_sha:
            raise Wwc1Error("completed joint composition source binding drifted")
        verify_embedded_facts(prior)
        return prior
    preflight = storage_preflight("compose", COMPOSE_PROJECTED_BYTES)
    oe1_result = run_oe1()
    binding = source_binding()
    field_fact, field_metadata = _materialize_joint_union(oe1_result)
    encode = bhw2.run_replay(
        family="jf2",
        root=STORE / "compose/jf2_joint_reencode",
        target_path=Path(field_fact["path"]),
        binding=binding,
        retain_argmax=False,
    )
    repeat = bhw2.run_replay(
        family="jf2",
        root=STORE / "compose/jf2_joint_reencode_repeat",
        target_path=Path(field_fact["path"]),
        binding=binding,
        retain_argmax=False,
    )
    stream = encode["streams"]["jf2_k060000"]
    repeat_stream = repeat["streams"]["jf2_k060000"]
    stream_identity = {key: stream[key] for key in ("bytes", "sha256")}
    repeat_stream_identity = {key: repeat_stream[key] for key in ("bytes", "sha256")}
    if stream_identity != repeat_stream_identity:
        raise Wwc1Error("joint JF2 re-encode deterministic repeat drifted")
    candidate_archive = bhw2.pack_stream_on_archive(
        JF2_ARCHIVE,
        Path(stream["path"]),
        STORE / "compose/retained/joint_union_on_jf2_k060.archive.zip",
    )
    repeat_archive = bhw2.pack_stream_on_archive(
        JF2_ARCHIVE,
        Path(repeat_stream["path"]),
        STORE / "compose/retained/joint_union_on_jf2_k060.repeat.archive.zip",
    )
    if candidate_archive["sha256"] != repeat_archive["sha256"] or candidate_archive["bytes"] != repeat_archive["bytes"]:
        raise Wwc1Error("joint candidate archive deterministic repeat drifted")
    decode = bhw2.verify_jf2_decode(
        archive=Path(candidate_archive["path"]),
        expected_tokens=Path(field_fact["path"]),
        library=Path(encode["build"]["library"]["path"]),
        destination=STORE / "compose/retained/joint_union_on_jf2_k060.decoded.u8",
    )
    base_archive = file_fact(JF2_ARCHIVE)
    delta = int(candidate_archive["bytes"]) - int(base_archive["bytes"])
    result = {
        "schema": "ddm_wwc1_compose_result.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "preflight": preflight,
        "source_binding_sha256": sha256_json(binding),
        "composition_rule": "coordinate union applied once on JF2 k060, then one real full-field JF2 RC64 re-encode",
        "field": field_fact,
        "field_metadata": field_metadata,
        "base_archive": base_archive,
        "candidate_archive": candidate_archive,
        "repeat_archive": repeat_archive,
        "deterministic_repeat_identical": True,
        "receiver_decode": decode,
        "marginal_bytes_vs_jf2_base": delta,
        "rate_only_delta_s": delta * S_PER_BYTE,
        "real_bits_per_actual_edit": (
            8.0 * delta / field_metadata["actual_edits_on_jf2_body"]
            if field_metadata["actual_edits_on_jf2_body"]
            else None
        ),
        "d_seg": "UNMEASURED",
        "d_pose": "UNMEASURED",
        "net_delta_s": "UNMEASURED",
        "scorer_ran_here": False,
        "seal_created_here": False,
    }
    atomic_json(result_path, result)
    return result


def write_manifest() -> dict[str, Any]:
    artifacts = [
        file_fact(path) for path in sorted(STORE.rglob("*")) if path.is_file() and path.name != "MANIFEST.json"
    ]
    result = {
        "schema": "ddm_wwc1_manifest.v1",
        "root": str(STORE),
        "artifact_count": len(artifacts),
        "artifact_bytes": sum(int(fact["bytes"]) for fact in artifacts),
        "artifacts": artifacts,
        "retention": "No listed payload may be moved or deleted without replacement custody and a reproducibility manifest.",
    }
    atomic_json(STORE / "MANIFEST.json", result)
    return result


def summarize() -> dict[str, Any]:
    control = run_control()
    oe1_result = run_oe1()
    compose = run_compose()
    bhw2_result = json.loads(BHW2_RESULT.read_text())
    fcd2_text = FCD2_MEMO.read_text()
    fcd3_text = FCD3_MEMO.read_text()
    if "0.00027348054805362656" not in fcd2_text or "0.0019433243907622244" not in fcd3_text:
        raise Wwc1Error("fresh FCD2/FCD3 baseline values were not found at source")
    rows = [
        {
            "family": "FCD1/DX2",
            "rows_screened": 1,
            "B": control["counts"]["B"],
            "H": control["counts"]["H"],
            "W": control["counts"]["W"],
            "real_archive_delta_bytes": -3_756,
            "real_bits_per_edit": -5.703872437,
            "distortion_disposition": "FCD2 exact union pose-refused; FCD3 pose-screened subset net +0.0019433243907622244 advisory",
        },
        {
            "family": "JF2",
            "scope": "1/7 registered rows (k060 optimal retained winner)",
            "rows_screened": 1,
            "B": bhw2_result["screen"]["counts"]["B"],
            "H": bhw2_result["screen"]["counts"]["H"],
            "W": bhw2_result["screen"]["counts"]["W"],
            "real_archive_delta_bytes": bhw2_result["rows"][0]["marginal_bytes_vs_own_base"],
            "real_bits_per_edit": bhw2_result["rows"][0]["real_bits_per_edit"],
            "distortion_disposition": "MAIN refused scorer fire after FCD2/FCD3 class-level adverse realization",
        },
        {
            "family": "LD1",
            "scope": "6/6 rows screened; values below are the family-induced support",
            "rows_screened": 6,
            "B": 14,
            "H": 0,
            "W": 0,
            "real_archive_delta_bytes": 1,
            "real_bits_per_edit": 0.571428571,
            "distortion_disposition": "WIN-WIN-VERIFIED-CLOSED at family-induced scope",
        },
        *[
            {
                "family": "OE1",
                "row": row["row"],
                "rows_screened": row["rows_screened"],
                "B": row["B"],
                "H": row["H"],
                "W": row["W"],
                "real_archive_delta_bytes": row["marginal_bytes_vs_own_base"],
                "real_bits_per_edit": row["real_bits_per_edit"],
                "distortion_disposition": "UNMEASURED; no scorer ownership",
            }
            for row in oe1_result["rows"]
        ],
        {
            "family": "DG2",
            "scope": "2/2 rows contained in JF2; no duplicate replay",
            "rows_screened": 0,
            "B": "CLOSED-BY-RECALL",
            "H": "CLOSED-BY-RECALL",
            "W": "CLOSED-BY-RECALL",
            "real_archive_delta_bytes": "CLOSED-BY-RECALL",
            "real_bits_per_edit": "CLOSED-BY-RECALL",
            "distortion_disposition": "DG2 k040/k060 are contained in the JF2 family; duplicate rerun forbidden",
        },
        {
            "family": "AE1",
            "scope": "physical-object prerequisite",
            "rows_screened": 0,
            "B": "ABSENT",
            "H": "ABSENT",
            "W": "ABSENT",
            "real_archive_delta_bytes": "ABSENT",
            "real_bits_per_edit": "ABSENT",
            "distortion_disposition": "no physical RC64 candidate or final coding-argmax object",
        },
    ]
    result = {
        "schema": "ddm_wwc1_result.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "control": file_fact(CONTROL_RESULT_PATH),
        "oe1": file_fact(STORE / "oe1/OE1_RESULT.json"),
        "compose": file_fact(STORE / "compose/COMPOSE_RESULT.json"),
        "rows": rows,
        "denominator": {
            "trade_space_families_enumerated": 5,
            "families_screened": 3,
            "families_closed_by_recall": 1,
            "families_absent_physical_object": 1,
            "screened": ["JF2", "LD1", "OE1"],
            "closed_by_recall": ["DG2 (contained in JF2)"],
            "absent": ["AE1 (no physical RC64/final coding-argmax object)"],
        },
        "fcd1_union_freshness": {
            "compensated_d_pose": 0.00027348054805362656,
            "same_instrument_base_d_pose": 0.0000063656845167356244,
            "pose_ratio": 42.96168736207959,
            "realized_d_seg_exact_union": "NOT-MEASURED; publish pose gate refused before full scorer",
            "pose_screened_subset_delta_s_advisory": 0.0019433243907622244,
            "net_delta_s_against_live_contest_cuda_pointer": "NOT-COMPARABLE ACROSS AXES",
        },
        "joint_composition": compose,
        "dual_axis_fire_order": "NOT-SEALED: no scorer-owned same-axis net row clears the admit bar",
        "scorer_ran_here": False,
        "modal_ran_here": False,
        "pointer_moved": False,
    }
    atomic_json(STORE / "RESULT.json", result)
    bhw2.write_jsonl(STORE / "ROWS.jsonl", rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "control", "oe1", "compose", "summarize", "manifest"))
    args = parser.parse_args()
    atomic_json(
        STORE / "launches" / f"{args.stage}.json",
        {
            "schema": "ddm_wwc1_launch.v1",
            "stage": args.stage,
            "semantic_argv": ["experiments/ddm_wwc1_winwin_cone_sweep.py", args.stage],
            "cwd": str(Path.cwd().resolve()),
            "python": sys.executable,
            "rng": "none used",
            "resume": "stage receipts and source bindings are validated before continuation",
            "write_tier": str(STORE),
            "scorer": "not loaded",
            "modal": "not invoked",
        },
    )
    if args.stage == "preflight":
        result = {
            "control": storage_preflight("control", CONTROL_PROJECTED_BYTES),
            "oe1_base": storage_preflight("oe1_base", OE1_BASE_PROJECTED_BYTES),
            "source_binding": source_binding(),
        }
    elif args.stage == "control":
        result = run_control()
    elif args.stage == "oe1":
        result = run_oe1()
    elif args.stage == "compose":
        result = run_compose()
    elif args.stage == "summarize":
        result = summarize()
    else:
        result = write_manifest()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
