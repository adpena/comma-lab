#!/usr/bin/env python3
"""Materialize and screen JF2/OE1 model-matched final coding argmax fields.

This is a scorer-free byte instrument.  It replays each family's *own* final
coding rows on the full n600 receiver trajectory, persists the final coding
argmax at every position, classifies the exact B/H/W partition against the
pinned DALI-lineage GT, and prices every GT-benefit field through a real RC64
re-encode.  It never substitutes the DX2 argmax for JF2 and never infers a
SegNet or PoseNet result from token labels.

All payloads and every 20-frame checkpoint are retained below APDataStore.
The two families are strictly sequential: ``oe1`` refuses unless JF2 reached a
typed byte refusal.  A negative-byte B cone instead emits a MAIN-owned scorer
fire order and terminates the sequence.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_df1_drop_field as df1
from experiments import ddm_fcd1_field_for_coder_diagonal as fcd1
from experiments import ddm_jg2_tail_reencode as jg2
from experiments import ddm_oe1_online_escape_member as oe1

AP_ROOT = Path("/Volumes/APDataStore/pact")
STORE = AP_ROOT / "ddm_bhw2_jf2_oe1_argmax_screen"
JF2_ROOT = AP_ROOT / "ddm_jf2_terminal_diagonal_harvest/retained/k060000"
JF2_ARCHIVE = JF2_ROOT / "retained/candidate_archive.zip"
JF2_RUNTIME = JF2_ROOT / "retained/candidate_runtime"
JF2_TOKENS = JF2_ROOT / "retained/decoded_tokens.u8"
JF2_MODEL = JF2_ROOT / "retained/model/hpac.ihs1.br.q10"
JF2_RESULT = JF2_ROOT / "MEASURE_RESULT.json"
OE1_ROOT = REPO / ".omx/tmp/arm_receipts_local/ddm_oe1_online_escape_member"
GT = Path("/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy")

CHARTER = REPO / ".omx/research/charters/ddm_bhw2_jf2_oe1_argmax_screen_20260829.md"
COMMON = REPO / ".omx/tmp/codex_runs/_common_contract.md"
BHW1_MEMO = REPO / ".omx/research/ddm_bhw1_winwin_cone_rescreen_20260829.md"
BHW1_DRIVER = REPO / "experiments/ddm_bhw1_winwin_cone_rescreen.py"
BHW1_RESULT = AP_ROOT / "ddm_bhw1_winwin_cone_rescreen/REAL_REENCODE_RESULT.json"
BHW1_MANIFEST = AP_ROOT / "ddm_bhw1_winwin_cone_rescreen/MANIFEST.json"

N, HEIGHT, WIDTH = 600, 384, 512
PLANE = HEIGHT * WIDTH
POSITIONS = N * PLANE
CLASSES = 5
STAGE_FRAMES = 20
RANK1_PROJECTED_BYTES = 2_025_181_467
RANK2_ARGMAX_FLOOR_BYTES = 589_824_000
RANK2_PROJECTED_BYTES = 5 << 30
RESERVE_BYTES = 8 << 30
LD1_SHARE = 0.00008325
AXIS = "[macOS-CPU frozen-scorer advisory]"
S_PER_BYTE = 25.0 / 37_545_489.0
REPLAY_SCHEMA = "ddm_bhw2_replay_stage.v1"
SCREEN_SCHEMA = "ddm_bhw2_screen_stage.v1"
OE1_DECODE_SCHEMA = "ddm_bhw2_oe1_decode_stage.v1"

OE1_SOURCE_PINS = {
    "control_w0": {
        "stream": (
            113_777,
            "e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5",
        ),
        "member": (
            180_268,
            "365f1b8d70463b250a2fe95e3599318ac90b31875cce5d66a767819404431c7a",
        ),
    },
    "escape_w1": {
        "stream": (
            126_082,
            "3c9996561b50b12e59fd7cb24f225434506f0f5d292ac966a16c46b35d171d99",
        ),
        "member": (
            192_573,
            "21728e895ecd9111c90fc09766c3058e5bd5401585aa2e02827129681b304911",
        ),
    },
    "escape_w4": {
        "stream": (
            125_509,
            "d37806b03736e9e86cb874be753db6dc196e7cad5c09272c8b2bc4fa59f00a68",
        ),
        "member": (
            192_000,
            "583dd80139ee94dfaecf6452998dba1489aa088ac3c76ab113b43e203a324544",
        ),
    },
    "escape_w16": {
        "stream": (
            124_898,
            "c8d4f445b507147997e1d802c446113eb4ed4deed339a0933131d88dc0ce190d",
        ),
        "member": (
            191_389,
            "de427eafff180542ec68e7bc10f1cef0820cd3a1afee60af0a84a0a3f02f7cef",
        ),
    },
    "escape_w64": {
        "stream": (
            124_595,
            "1f45a4bdd59dfec6e75c5ed052e2e26600514e3ed461abd73924e92f6ae2ef3b",
        ),
        "member": (
            191_086,
            "cec655f5715bd6b1174c8d01a8d799cf1d934f9e1ae1951791c24d1525ada4af",
        ),
    },
}

PINS: dict[str, tuple[Path, int, str]] = {
    "charter": (CHARTER, 0, ""),
    "common": (COMMON, 0, ""),
    "bhw1_memo": (
        BHW1_MEMO,
        0,
        "71579ac540a36f65eb5e8707b2896b2eafb001ae8ee0c7774d8f505408d8b55a",
    ),
    "bhw1_driver": (
        BHW1_DRIVER,
        0,
        "bf31e6043b0524efaafead0a663e66743da33b4a4f5ae4e81f96d5c772fe072b",
    ),
    "bhw1_result": (
        BHW1_RESULT,
        3_282,
        "a7e87777b5b7fe66fe67680e3c7b83e17f29abc4c05797032dcbed31f9227f48",
    ),
    "bhw1_manifest": (
        BHW1_MANIFEST,
        59_826,
        "5bd824f90616f29fbb89e00d0ff46a3103f6d3db402da6a2dcb56ae9c2716c9e",
    ),
    "jf2_archive": (
        JF2_ARCHIVE,
        178_792,
        "59428f07e6344129d2c5e37ffac84ec19f8e609b2b5951d0d970fb694b88c54a",
    ),
    "jf2_tokens": (
        JF2_TOKENS,
        POSITIONS,
        "15018481bd8007dd9099d1b67d5e8014283465d062a34ba3f06b3450758b5878",
    ),
    "jf2_model": (
        JF2_MODEL,
        13_398,
        "98b96ee585f16250b14a05c2202c67541f7717e01c63dfbf068f0af7a714ddc0",
    ),
    "gt": (
        GT,
        117_964_928,
        "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248",
    ),
    "oe1_tokens": (
        oe1.TOKENS,
        POSITIONS,
        "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb",
    ),
}


class Bhw2Error(RuntimeError):
    """A source, storage, resume, receiver, or sequencing gate refused."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_fact(fact: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(fact["path"]))
    observed = file_fact(path)
    if observed != dict(fact):
        raise Bhw2Error(f"artifact drifted: expected={dict(fact)}, observed={observed}")
    return observed


def require_pin(key: str) -> dict[str, Any]:
    path, size, digest = PINS[key]
    if not path.is_file():
        raise Bhw2Error(f"pinned source is absent: {key}: {path}")
    observed = file_fact(path)
    if size and observed["bytes"] != size:
        raise Bhw2Error(f"pinned source size drifted: {key}: {observed}")
    if digest and observed["sha256"] != digest:
        raise Bhw2Error(f"pinned source sha drifted: {key}: {observed}")
    return observed


def atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_fact(path)


def atomic_json(path: Path, payload: object) -> dict[str, Any]:
    return atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def atomic_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_fact(path)


def atomic_npz(path: Path, **values: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_fact(path)


def source_binding() -> dict[str, Any]:
    required = {key: require_pin(key) for key in PINS}
    oe1_rows: dict[str, Any] = {}
    for window in oe1.WINDOWS:
        rung = oe1.label(window)
        root = OE1_ROOT / "retained/rungs" / rung
        observed = {
            "stream": file_fact(root / "tokens.rc64"),
            "member": file_fact(root / "member.bin"),
            "decoded_tokens": file_fact(root / "decoded_tokens.u8"),
        }
        for kind in ("stream", "member"):
            expected_bytes, expected_sha = OE1_SOURCE_PINS[rung][kind]
            if observed[kind]["bytes"] != expected_bytes or observed[kind]["sha256"] != expected_sha:
                raise Bhw2Error(f"OE1 pinned {rung}/{kind} drifted: {observed[kind]}")
        if (
            observed["decoded_tokens"]["bytes"] != POSITIONS
            or observed["decoded_tokens"]["sha256"] != PINS["oe1_tokens"][2]
        ):
            raise Bhw2Error(f"OE1 pinned {rung} decoded field drifted: {observed['decoded_tokens']}")
        oe1_rows[rung] = observed
    implementation = file_fact(Path(__file__))
    return {
        "schema": "ddm_bhw2_source_binding.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "shape": [N, HEIGHT, WIDTH],
        "required_pins": required,
        "jf2_result": file_fact(JF2_RESULT),
        "jf2_runtime_archive": file_fact(JF2_RUNTIME / "archive.zip"),
        "oe1_result": file_fact(OE1_ROOT / "RESULT.json"),
        "oe1_manifest": file_fact(OE1_ROOT / "MANIFEST.json"),
        "oe1_rows": oe1_rows,
        "implementation": implementation,
    }


def storage_preflight(phase: str, projected_bytes: int) -> dict[str, Any]:
    STORE.mkdir(parents=True, exist_ok=True)
    if not STORE.resolve().is_relative_to(AP_ROOT.resolve()):
        raise Bhw2Error(f"bulk store escaped APDataStore: {STORE}")
    free = shutil.disk_usage(STORE).free
    required = projected_bytes + RESERVE_BYTES
    result = {
        "schema": "ddm_bhw2_storage_preflight.v1",
        "phase": phase,
        "path": str(STORE),
        "free_bytes_before": free,
        "projected_materialization_bytes": projected_bytes,
        "reserve_bytes": RESERVE_BYTES,
        "required_bytes": required,
        "status": "PASS" if free >= required else "BLOCKED_SHORTFALL",
        "shortfall_bytes": max(0, required - free),
        "cleanup_attempted": False,
    }
    atomic_json(STORE / "preflight" / f"{phase}.json", result)
    if free < required:
        raise Bhw2Error(
            f"storage preflight blocked {phase}: free={free}, required={required}, "
            f"shortfall={required - free}; no deletion attempted"
        )
    return result


def purge_runtime_modules() -> None:
    for name in list(sys.modules):
        if name in {"runtime", "cpr1"} or name.startswith(("runtime.", "cpr1.")):
            del sys.modules[name]


def stage_root(root: Path, start: int, end: int) -> Path:
    return root / "stages" / f"frames_{start:04d}_{end - 1:04d}"


def contiguous_receipts(root: Path, *, schema: str, binding_sha256: str) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for start in range(0, N, STAGE_FRAMES):
        end = min(start + STAGE_FRAMES, N)
        path = stage_root(root, start, end) / "RECEIPT.json"
        if not path.is_file():
            break
        row = json.loads(path.read_text())
        if (
            row.get("schema") != schema
            or int(row.get("frame_start", -1)) != start
            or int(row.get("frame_end", -1)) != end
            or row.get("execution_binding_sha256") != binding_sha256
        ):
            raise Bhw2Error(f"stage receipt schema/span/binding drifted: {path}")
        for fact in row["artifacts"].values():
            verify_fact(fact)
        receipts.append(row)
    all_receipts = list((root / "stages").glob("frames_*/RECEIPT.json"))
    if len(all_receipts) != len(receipts):
        raise Bhw2Error(f"stage receipts are not a contiguous prefix below {root}")
    return receipts


class Jf2Adapter:
    """The exact JF2 k060 refit model plus native JG2 RC64 encoder."""

    names = ("jf2_k060000",)

    def __init__(self, root: Path, tag: str) -> None:
        purge_runtime_modules()
        env = jg2._prepare(SimpleNamespace(store=str(root), runtime_root=str(JF2_RUNTIME)), tag)
        torch = importlib.import_module("torch")
        free_corrector = importlib.import_module("runtime.free_corrector")
        hpac_inference = importlib.import_module("runtime.hpac_inference")
        device = torch.device("cpu")
        residual = env["residual"]
        renderer = env["renderer"]
        parts = env["parts"]
        base_hpac = residual.materialize_ihs1(parts.hpac_blob, renderer)
        model = renderer.load_hpac(base_hpac, device)
        sparse = residual._sparse_class(env["renderer_dir"])(model, HEIGHT, WIDTH)
        hpac_inference.optimize_sparse_evaluator(sparse)
        plans = []
        for mask in renderer.group_masks(device):
            flat = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1)).astype(np.int64)
            plans.append((torch.from_numpy(flat).to(device), flat))
        self.runtime = {
            "torch": torch,
            "device": device,
            "residual": residual,
            "renderer": renderer,
            "parts": parts,
            "model": model,
            "sparse": sparse,
            "plans": plans,
            "corrector": free_corrector.FreeCorrector(PLANE),
            "cold": free_corrector.FreeCorrector(PLANE),
        }
        self.library = env["library"]
        self.route_b = env["route_b"]
        self.build = env["build"]

    def new_encoders(self) -> dict[str, Any]:
        return {self.names[0]: self.route_b.NativeRc64Encoder(self.library)}

    def resume_encoders(self, root: Path) -> dict[str, Any]:
        state = (root / f"encoder_{self.names[0]}.bin").read_bytes()
        return {self.names[0]: self.route_b.NativeRc64Encoder(self.library, state)}

    def snapshot(self, encoder: Any) -> bytes:
        return encoder.snapshot()

    def finish(self, encoder: Any) -> bytes:
        payload = encoder.finish()
        if not payload.startswith(self.route_b.TOKEN_MAGIC):
            raise Bhw2Error("JF2 RC64 payload lost its magic")
        size = int(encoder.library.rc64_encoder_size(encoder.context))
        pointer = encoder.library.rc64_encoder_data(encoder.context)
        if not size or not pointer:
            raise Bhw2Error("JF2 RC64 encoder produced no payload")
        return ctypes.string_at(pointer, size)

    def candidate_rows(
        self,
        base: np.ndarray,
        _group: int,
        _feature: np.ndarray,
        _symbols: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], object]:
        return {self.names[0]: base}, None

    def observe(self, _frame: int, _group: int, _feature: np.ndarray, _value: object) -> None:
        return

    def argmax(self, coding: np.ndarray) -> np.ndarray:
        winner, _top, _second = df1.coding_prediction(coding)
        return winner

    def extra_state(self) -> dict[str, np.ndarray]:
        return {}

    def restore_extra(self, _payload: Any) -> None:
        return


class Oe1Adapter:
    """The exact OE1 online-mixture family over a caller-selected rung set."""

    def __init__(self, root: Path, windows: Iterable[int]) -> None:
        self.windows = tuple(int(value) for value in windows)
        self.names = tuple(oe1.label(value) for value in self.windows)
        purge_runtime_modules()
        library, rc64 = oe1.compile_rc64(root)
        self.runtime = oe1.load_receiver(library)
        self.library = library
        self.rc64 = rc64
        self.states = {window: oe1.EscapeState(window) for window in self.windows}
        build_path = root / "work/RC64_BUILD.json"
        self.build = {
            **json.loads(build_path.read_text()),
            "receipt": file_fact(build_path),
        }

    def new_encoders(self) -> dict[str, Any]:
        return {oe1.label(window): self.rc64.NativeEncoder(self.library) for window in self.windows}

    def resume_encoders(self, root: Path) -> dict[str, Any]:
        return {
            oe1.label(window): oe1.cp._rc64_resume(
                self.rc64.NativeEncoder,
                self.library,
                (root / f"encoder_{oe1.label(window)}.bin").read_bytes(),
            )
            for window in self.windows
        }

    def snapshot(self, encoder: Any) -> bytes:
        return oe1.cp._rc64_snapshot(encoder)

    def finish(self, encoder: Any) -> bytes:
        return encoder.finish()

    def candidate_rows(
        self,
        base: np.ndarray,
        group: int,
        feature: np.ndarray,
        symbols: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], object]:
        frequencies, _costs = oe1.selected_costs(self.rc64, base, symbols)
        selected = frequencies[np.arange(symbols.size), symbols]
        anti = selected.astype(np.uint64) * CLASSES < oe1.TOTAL
        rows = {oe1.label(window): self.states[window].coding(base, group, feature) for window in self.windows}
        return rows, anti

    def observe(self, frame: int, group: int, feature: np.ndarray, value: object) -> None:
        anti = np.asarray(value, dtype=bool)
        for window in self.windows:
            self.states[window].observe(frame, group, feature, anti)

    def argmax(self, coding: np.ndarray) -> np.ndarray:
        winner, _top, _second = df1.coding_prediction(coding)
        return winner

    def extra_state(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for window in self.windows:
            arrays.update(self.states[window].arrays(oe1.label(window)))
        return arrays

    def restore_extra(self, payload: Any) -> None:
        for window in self.windows:
            self.states[window].load(payload, oe1.label(window))


def replay_state_arrays(adapter: Any, frame_end: int, previous: np.ndarray) -> dict[str, np.ndarray]:
    corrector = adapter.runtime["corrector"]
    captured = jg2.corrector_state(corrector)
    lost = jg2.uncaptured_divergent_state(corrector, adapter.runtime["cold"], set(captured))
    if lost:
        raise Bhw2Error(f"checkpoint would lose corrector state: {lost[:8]}")
    return {
        "schema": np.frombuffer(REPLAY_SCHEMA.encode(), dtype=np.uint8),
        "frame_end": np.asarray([frame_end], dtype=np.int64),
        "previous": np.asarray(previous, dtype=np.uint8),
        **{f"corrector__{key}": value for key, value in captured.items()},
        **adapter.extra_state(),
    }


def restore_replay_state(adapter: Any, path: Path) -> tuple[int, Any]:
    torch = adapter.runtime["torch"]
    with np.load(path, allow_pickle=False) as payload:
        if bytes(payload["schema"]).decode() != REPLAY_SCHEMA:
            raise Bhw2Error(f"replay checkpoint schema drifted: {path}")
        frame_end = int(payload["frame_end"][0])
        corrector = {
            key.removeprefix("corrector__"): payload[key].copy()
            for key in payload.files
            if key.startswith("corrector__")
        }
        jg2.load_corrector_state(adapter.runtime["corrector"], corrector)
        adapter.restore_extra(payload)
        previous_np = np.asarray(payload["previous"], dtype=np.uint8).copy()
    previous = torch.from_numpy(previous_np.astype(np.int64)).reshape(1, HEIGHT, WIDTH).to(adapter.runtime["device"])
    return frame_end, previous


def run_replay(
    *,
    family: str,
    root: Path,
    target_path: Path,
    binding: dict[str, Any],
    retain_argmax: bool,
    windows: Iterable[int] = (),
    expected_streams: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    result_path = root / "REPLAY_RESULT.json"
    target_fact = file_fact(target_path)
    source_binding_sha = sha256_json(binding)
    execution_binding = {
        "source_binding_sha256": source_binding_sha,
        "family": family,
        "target": target_fact,
        "retain_argmax": retain_argmax,
        "windows": tuple(int(value) for value in windows),
    }
    binding_sha = sha256_json(execution_binding)
    if result_path.is_file():
        prior = json.loads(result_path.read_text())
        if prior.get("execution_binding_sha256") != binding_sha:
            raise Bhw2Error(f"completed replay source binding drifted: {result_path}")
        for fact in prior["streams"].values():
            verify_fact(fact)
        for fact in prior.get("coding_argmax", {}).values():
            verify_fact(fact)
        return prior

    adapter: Any
    if family == "jf2":
        adapter = Jf2Adapter(root, root.name)
    elif family == "oe1":
        adapter = Oe1Adapter(root, windows)
    else:
        raise Bhw2Error(f"unknown replay family: {family}")
    names = tuple(adapter.names)
    receipts = contiguous_receipts(root, schema=REPLAY_SCHEMA, binding_sha256=binding_sha)
    if receipts:
        last = receipts[-1]
        last_root = stage_root(root, int(last["frame_start"]), int(last["frame_end"]))
        start_frame, previous = restore_replay_state(adapter, last_root / "receiver_state.npz")
        encoders = adapter.resume_encoders(last_root)
    else:
        start_frame = 0
        torch = adapter.runtime["torch"]
        previous = torch.zeros((1, HEIGHT, WIDTH), dtype=torch.long, device=adapter.runtime["device"])
        encoders = adapter.new_encoders()

    target = np.memmap(target_path, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    runtime = adapter.runtime
    torch = runtime["torch"]
    residual = runtime["residual"]
    renderer = runtime["renderer"]
    model = runtime["model"]
    sparse = runtime["sparse"]
    corrector = runtime["corrector"]
    parts = runtime["parts"]
    device = runtime["device"]
    started = time.perf_counter()

    with torch.inference_mode():
        for stage_start in range(start_frame, N, STAGE_FRAMES):
            stage_end = min(stage_start + STAGE_FRAMES, N)
            argmax_stage = (
                {name: np.empty((stage_end - stage_start, HEIGHT, WIDTH), dtype=np.uint8) for name in names}
                if retain_argmax
                else {}
            )
            for frame in range(stage_start, stage_end):
                offset = frame - stage_start
                current = torch.zeros_like(previous)
                index = torch.tensor([frame], dtype=torch.long, device=device)
                context = model.prepare_frame_context(index, previous)
                if frame:
                    previous_cpu = previous[0].to(device="cpu", dtype=torch.uint8).numpy()
                    boundary = residual._boundary_buckets(previous_cpu).reshape(-1)
                else:
                    boundary = np.full(PLANE, 4, dtype=np.uint8)
                corrector.begin_frame(boundary)
                plane_target = np.asarray(target[frame]).reshape(-1)
                for group, (device_positions, flat_positions) in enumerate(runtime["plans"]):
                    base_logits = sparse.selected_logits(current, context, group).cpu().numpy()
                    predicted = base_logits.argmax(axis=1).astype(np.int64)
                    feature = boundary[flat_positions].astype(np.int64) * CLASSES + predicted
                    corrected = base_logits + parts.table.values[feature]
                    probability = residual._probability_table(corrected, renderer.HPAC_LOGIT_PRECISION)
                    receiver_state = corrector.group_state(probability, predicted, flat_positions)
                    base = np.asarray(corrector.coding_row(receiver_state), dtype=np.float32)
                    symbols = plane_target[flat_positions].astype(np.int64)
                    rows, observation = adapter.candidate_rows(base, group, feature, symbols)
                    for name, coding in rows.items():
                        encoders[name].encode(symbols.astype(np.int32), coding)
                        if retain_argmax:
                            argmax_stage[name][offset].reshape(-1)[flat_positions] = adapter.argmax(coding)
                    adapter.observe(frame, group, feature, observation)
                    corrector.observe(receiver_state, symbols)
                    current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(device)
                frame_tokens = current[0].to(device="cpu", dtype=torch.uint8).numpy()
                if not np.array_equal(frame_tokens.reshape(-1), plane_target):
                    raise Bhw2Error(f"{family} replay diverged from target at frame {frame}")
                corrector.end_frame(frame_tokens.reshape(-1))
                previous = current

            current_root = stage_root(root, stage_start, stage_end)
            artifacts: dict[str, dict[str, Any]] = {}
            for name in names:
                artifacts[f"encoder_{name}"] = atomic_bytes(
                    current_root / f"encoder_{name}.bin",
                    adapter.snapshot(encoders[name]),
                )
                if retain_argmax:
                    artifacts[f"argmax_{name}"] = atomic_npy(current_root / f"argmax_{name}.u8.npy", argmax_stage[name])
            arrays = replay_state_arrays(
                adapter,
                stage_end,
                previous[0].to(device="cpu", dtype=torch.uint8).numpy(),
            )
            artifacts["receiver_state"] = atomic_npz(current_root / "receiver_state.npz", **arrays)
            receipt = {
                "schema": REPLAY_SCHEMA,
                "source_binding_sha256": source_binding_sha,
                "execution_binding_sha256": binding_sha,
                "family": family,
                "frame_start": stage_start,
                "frame_end": stage_end,
                "retain_argmax": retain_argmax,
                "target": target_fact,
                "artifacts": artifacts,
                "elapsed_seconds": time.perf_counter() - started,
            }
            atomic_json(current_root / "RECEIPT.json", receipt)
            receipts.append(receipt)
            print(
                json.dumps(
                    {
                        "family": family,
                        "replay": root.name,
                        "frame_end": stage_end,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if len(receipts) != N // STAGE_FRAMES:
        raise Bhw2Error(f"{family} replay did not retain all 30 stage checkpoints")
    streams = {
        name: atomic_bytes(root / "retained/streams" / f"{name}.rc64", adapter.finish(encoder))
        for name, encoder in encoders.items()
    }
    stream_identity: dict[str, bool] = {}
    if expected_streams is not None:
        for name, expected in expected_streams.items():
            stream_identity[name] = streams[name]["bytes"] == int(expected["bytes"]) and streams[name]["sha256"] == str(
                expected["sha256"]
            )
            if not stream_identity[name]:
                raise Bhw2Error(
                    f"{family} regenerated stream differs from its retained source: "
                    f"name={name}, expected={dict(expected)}, observed={streams[name]}"
                )
    argmax_facts = assemble_argmax(root, names) if retain_argmax else {}
    result = {
        "schema": "ddm_bhw2_replay_result.v1",
        "complete": True,
        "family": family,
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "target": target_fact,
        "source_binding_sha256": source_binding_sha,
        "execution_binding_sha256": binding_sha,
        "stages": len(receipts),
        "streams": streams,
        "stream_identity_to_family_source": stream_identity,
        "coding_argmax": argmax_facts,
        "build": adapter.build,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(result_path, result)
    return result


def assemble_argmax(root: Path, names: Iterable[str]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name in names:
        destination = root / "retained/argmax" / f"{name}.coding_argmax.u8.bin"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f".partial.{os.getpid()}")
        digest = hashlib.sha256()
        written = 0
        try:
            with temporary.open("wb") as handle:
                for start in range(0, N, STAGE_FRAMES):
                    end = min(start + STAGE_FRAMES, N)
                    value = np.load(
                        stage_root(root, start, end) / f"argmax_{name}.u8.npy",
                        allow_pickle=False,
                    )
                    payload = np.ascontiguousarray(value, dtype=np.uint8).tobytes()
                    handle.write(payload)
                    digest.update(payload)
                    written += len(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        if written != POSITIONS or sha256_file(destination) != digest.hexdigest():
            raise Bhw2Error(f"assembled argmax failed size/hash closure: {destination}")
        output[name] = file_fact(destination)
    return output


def screen_stage_root(root: Path, start: int, end: int) -> Path:
    return root / "stages" / f"frames_{start:04d}_{end - 1:04d}"


def classify_and_materialize(
    *,
    row: str,
    root: Path,
    tokens_path: Path,
    argmax_path: Path,
    binding: dict[str, Any],
) -> dict[str, Any]:
    result_path = root / "SCREEN_RESULT.json"
    token_fact = file_fact(tokens_path)
    argmax_fact = file_fact(argmax_path)
    gt_fact = file_fact(GT)
    source_binding_sha = sha256_json(binding)
    execution_binding = {
        "source_binding_sha256": source_binding_sha,
        "row": row,
        "tokens": token_fact,
        "coding_argmax": argmax_fact,
        "gt": gt_fact,
    }
    binding_sha = sha256_json(execution_binding)
    if result_path.is_file():
        prior = json.loads(result_path.read_text())
        if prior.get("execution_binding_sha256") != binding_sha:
            raise Bhw2Error(f"completed screen source binding drifted: {result_path}")
        for fact in prior["payloads"].values():
            verify_fact(fact)
        return prior
    tokens = np.memmap(tokens_path, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    argmax = np.memmap(argmax_path, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    gt = np.load(GT, mmap_mode="r", allow_pickle=False)
    if gt.shape != (N, HEIGHT, WIDTH) or gt.dtype != np.uint8:
        raise Bhw2Error(f"GT geometry drifted: shape={gt.shape}, dtype={gt.dtype}")
    counts = {"B": 0, "H": 0, "W": 0}
    started = time.perf_counter()
    for start in range(0, N, STAGE_FRAMES):
        end = min(start + STAGE_FRAMES, N)
        current_root = screen_stage_root(root, start, end)
        receipt_path = current_root / "RECEIPT.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("schema") != SCREEN_SCHEMA or receipt.get("execution_binding_sha256") != binding_sha:
                raise Bhw2Error(f"screen stage receipt drifted: {receipt_path}")
            for fact in receipt["artifacts"].values():
                verify_fact(fact)
            for key in counts:
                counts[key] += int(receipt["counts"][key])
            continue
        token_block = np.asarray(tokens[start:end])
        argmax_block = np.asarray(argmax[start:end])
        gt_block = np.asarray(gt[start:end])
        classified = fcd1.classify_pool(token_block, argmax_block, gt_block)
        labels = np.zeros(token_block.shape, dtype=np.uint8)
        labels[classified["benefit"]] = 1
        labels[classified["harm"]] = 2
        labels[classified["wash"]] = 3
        benefit = token_block.copy()
        benefit[classified["benefit"]] = argmax_block[classified["benefit"]]
        local_coords = np.argwhere(classified["benefit"]).astype(np.int32)
        if local_coords.size:
            local_coords[:, 0] += start
            index = tuple(np.moveaxis(np.argwhere(classified["benefit"]), 1, 0))
            old = token_block[index].astype(np.uint8)
            new = argmax_block[index].astype(np.uint8)
        else:
            old = np.empty(0, dtype=np.uint8)
            new = np.empty(0, dtype=np.uint8)
        stage_counts = {
            "B": int(classified["benefit"].sum()),
            "H": int(classified["harm"].sum()),
            "W": int(classified["wash"].sum()),
        }
        artifacts = {
            "labels": atomic_npy(current_root / "bhw_labels.u8.npy", labels),
            "benefit_field": atomic_npy(current_root / "benefit_field.u8.npy", benefit),
            "benefit_coordinates": atomic_npz(
                current_root / "benefit.frame_y_x_old_new.npz",
                coords=local_coords,
                old=old,
                new=new,
            ),
        }
        receipt = {
            "schema": SCREEN_SCHEMA,
            "source_binding_sha256": source_binding_sha,
            "execution_binding_sha256": binding_sha,
            "row": row,
            "frame_start": start,
            "frame_end": end,
            "counts": stage_counts,
            "artifacts": artifacts,
        }
        atomic_json(receipt_path, receipt)
        for key in counts:
            counts[key] += stage_counts[key]
    payloads = assemble_screen(root)
    disagreements = sum(counts.values())
    share = counts["B"] / disagreements if disagreements else 0.0
    result = {
        "schema": "ddm_bhw2_screen_result.v1",
        "complete": True,
        "row": row,
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "source_binding_sha256": source_binding_sha,
        "execution_binding_sha256": binding_sha,
        "population": "n600 full token lattice",
        "rows_screened": 600,
        "selection_rule": (
            "B: token!=coding_argmax and token!=GT and coding_argmax==GT; "
            "H: token!=coding_argmax and token==GT and coding_argmax!=GT; "
            "W: token!=coding_argmax and token!=GT and coding_argmax!=GT"
        ),
        "counts": counts,
        "disagreements": disagreements,
        "B_share_of_disagreements": share,
        "ld1_share": LD1_SHARE,
        "B_share_over_ld1": share / LD1_SHARE if LD1_SHARE else math.inf,
        "tokens": token_fact,
        "coding_argmax": argmax_fact,
        "gt": gt_fact,
        "payloads": payloads,
        "d_seg": "UNMEASURED",
        "d_pose": "UNMEASURED",
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(result_path, result)
    return result


def assemble_screen(root: Path) -> dict[str, dict[str, Any]]:
    destinations = {
        "bhw_labels": root / "retained/bhw_labels.u8.bin",
        "benefit_field": root / "retained/benefit_field.u8.bin",
    }
    output: dict[str, dict[str, Any]] = {}
    for key, destination in destinations.items():
        source_name = "bhw_labels.u8.npy" if key == "bhw_labels" else "benefit_field.u8.npy"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f".partial.{os.getpid()}")
        try:
            with temporary.open("wb") as handle:
                for start in range(0, N, STAGE_FRAMES):
                    end = min(start + STAGE_FRAMES, N)
                    value = np.load(
                        screen_stage_root(root, start, end) / source_name,
                        allow_pickle=False,
                    )
                    handle.write(np.ascontiguousarray(value, dtype=np.uint8).tobytes())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        if destination.stat().st_size != POSITIONS:
            raise Bhw2Error(f"assembled screen payload has wrong size: {destination}")
        output[key] = file_fact(destination)

    coords_parts: list[np.ndarray] = []
    old_parts: list[np.ndarray] = []
    new_parts: list[np.ndarray] = []
    for start in range(0, N, STAGE_FRAMES):
        end = min(start + STAGE_FRAMES, N)
        with np.load(
            screen_stage_root(root, start, end) / "benefit.frame_y_x_old_new.npz",
            allow_pickle=False,
        ) as payload:
            coords_parts.append(payload["coords"].copy())
            old_parts.append(payload["old"].copy())
            new_parts.append(payload["new"].copy())
    coords = np.concatenate(coords_parts, axis=0)
    old = np.concatenate(old_parts)
    new = np.concatenate(new_parts)
    output["benefit_coordinates"] = atomic_npz(
        root / "retained/benefit.frame_y_x_old_new.npz",
        coords=coords,
        old=old,
        new=new,
    )
    return output


def pack_stream_on_archive(base_archive: Path, stream_path: Path, destination: Path) -> dict[str, Any]:
    sections = jg2.split_member(jg2.read_archive_member(base_archive))
    if len(sections["tail"]) < jg2.RESIDUAL_COMPACT_BYTES:
        raise Bhw2Error(f"base archive tail is too short: {base_archive}")
    sections["tail"] = sections["tail"][: jg2.RESIDUAL_COMPACT_BYTES] + stream_path.read_bytes()
    jg2.pack_archive(jg2.join_member(sections), destination)
    return file_fact(destination)


def verify_jf2_decode(*, archive: Path, expected_tokens: Path, library: Path, destination: Path) -> dict[str, Any]:
    purge_runtime_modules()
    residual, renderer, renderer_dir = jg2.load_runtime(JF2_RUNTIME)
    previous = os.environ.get("CPR1_RC64_LIBRARY")
    os.environ["CPR1_RC64_LIBRARY"] = str(library)
    try:
        parts = residual.read_residual_archive(archive)
        decoded_tensor, report = residual.decode_production_tokens(
            parts, renderer, renderer_dir, importlib.import_module("torch").device("cpu")
        )
    finally:
        if previous is None:
            os.environ.pop("CPR1_RC64_LIBRARY", None)
        else:
            os.environ["CPR1_RC64_LIBRARY"] = previous
    decoded = decoded_tensor.numpy().astype(np.uint8, copy=False)
    decoded_fact = atomic_bytes(destination, decoded.tobytes(order="C"))
    expected = np.memmap(expected_tokens, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    identity = bool(np.array_equal(decoded, expected))
    if not identity:
        raise Bhw2Error("JF2 benefit archive failed exact production-token decode identity")
    return {"decoded_tokens": decoded_fact, "identity": identity, "report": report}


def assemble_oe1_decoded(root: Path, name: str) -> dict[str, Any]:
    destination = root / "retained/decoded" / f"{name}.decoded_tokens.u8"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".partial.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            for start in range(0, N, STAGE_FRAMES):
                end = min(start + STAGE_FRAMES, N)
                value = np.load(
                    stage_root(root, start, end) / f"decoded_{name}.u8.npy",
                    allow_pickle=False,
                )
                handle.write(np.ascontiguousarray(value, dtype=np.uint8).tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    if destination.stat().st_size != POSITIONS:
        raise Bhw2Error(f"assembled OE1 decoded field has wrong size: {destination}")
    return file_fact(destination)


def decode_oe1_streams(
    *,
    root: Path,
    target_path: Path,
    streams: Mapping[str, Mapping[str, Any]],
    windows: Iterable[int],
    binding: dict[str, Any],
) -> dict[str, Any]:
    selected_windows = tuple(int(value) for value in windows)
    names = tuple(oe1.label(value) for value in selected_windows)
    target_fact = file_fact(target_path)
    if target_fact["bytes"] != POSITIONS:
        raise Bhw2Error(f"OE1 decode target has wrong size: {target_fact}")
    stream_facts = {name: verify_fact(streams[name]) for name in names}
    decode_binding = {
        "source_binding_sha256": sha256_json(binding),
        "target": target_fact,
        "streams": stream_facts,
        "windows": selected_windows,
    }
    binding_sha = sha256_json(decode_binding)
    result_path = root / "DECODE_RESULT.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text())
        if prior.get("decode_binding_sha256") != binding_sha:
            raise Bhw2Error(f"completed OE1 decode binding drifted: {result_path}")
        for row in prior["rows"].values():
            verify_fact(row["decoded_tokens"])
        return prior

    purge_runtime_modules()
    library, rc64 = oe1.compile_rc64(root)
    runtime = oe1.load_receiver(library)
    states = {window: oe1.EscapeState(window) for window in selected_windows}
    decoders = {
        window: rc64.NativeDecoder(library, Path(stream_facts[oe1.label(window)]["path"]).read_bytes())
        for window in selected_windows
    }
    receipts = contiguous_receipts(root, schema=OE1_DECODE_SCHEMA, binding_sha256=binding_sha)
    if receipts:
        last = receipts[-1]
        last_root = stage_root(root, int(last["frame_start"]), int(last["frame_end"]))
        start_frame, previous = oe1.restore_common_state(
            runtime,
            states,
            last_root / "receiver_state.npz",
            OE1_DECODE_SCHEMA,
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
            decoded_stage = {
                window: np.empty((stage_end - stage_start, HEIGHT, WIDTH), dtype=np.uint8)
                for window in selected_windows
            }
            for frame in range(stage_start, stage_end):
                offset = frame - stage_start
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
                for group, (device_positions, flat_positions) in enumerate(runtime["plans"]):
                    base_logits = sparse.selected_logits(current, context, group).cpu().numpy()
                    predicted = base_logits.argmax(axis=1).astype(np.int64)
                    feature = boundary[flat_positions].astype(np.int64) * CLASSES + predicted
                    corrected = base_logits + parts.table.values[feature]
                    probability = residual._probability_table(corrected, runtime["renderer"].HPAC_LOGIT_PRECISION)
                    receiver_state = corrector.group_state(probability, predicted, flat_positions)
                    base = np.asarray(corrector.coding_row(receiver_state), dtype=np.float32)
                    expected = expected_plane[flat_positions].astype(np.int64)
                    base_frequency, _costs = oe1.selected_costs(rc64, base, expected)
                    selected = base_frequency[np.arange(expected.size), expected]
                    anti = selected.astype(np.uint64) * CLASSES < oe1.TOTAL
                    for window in selected_windows:
                        candidate = states[window].coding(base, group, feature)
                        decoded = decoders[window].decode(candidate).astype(np.int64)
                        if not np.array_equal(decoded, expected):
                            mismatch = int(np.flatnonzero(decoded != expected)[0])
                            raise Bhw2Error(
                                "OE1 candidate receiver mismatch: "
                                f"rung={oe1.label(window)}, frame={frame}, "
                                f"group={group}, within_group={mismatch}"
                            )
                        states[window].observe(frame, group, feature, anti)
                    corrector.observe(receiver_state, expected)
                    current.reshape(-1)[device_positions] = torch.from_numpy(expected).to(device)
                frame_tokens = current[0].to(device="cpu", dtype=torch.uint8).numpy()
                for window in selected_windows:
                    decoded_stage[window][offset] = frame_tokens
                corrector.end_frame(frame_tokens.reshape(-1))
                previous = current

            current_root = stage_root(root, stage_start, stage_end)
            artifacts: dict[str, dict[str, Any]] = {}
            for window in selected_windows:
                name = oe1.label(window)
                artifacts[f"decoded_{name}"] = atomic_npy(
                    current_root / f"decoded_{name}.u8.npy",
                    decoded_stage[window],
                )
            decoder_arrays = {oe1.label(window): oe1.bl1.decoder_state(decoders[window]) for window in selected_windows}
            artifacts["decoder_states"] = atomic_npz(current_root / "decoder_states.npz", **decoder_arrays)
            arrays = oe1.common_state_arrays(
                runtime,
                states,
                OE1_DECODE_SCHEMA,
                stage_end,
                previous[0].to(device="cpu", dtype=torch.uint8).numpy(),
            )
            artifacts["receiver_state"] = atomic_npz(current_root / "receiver_state.npz", **arrays)
            receipt = {
                "schema": OE1_DECODE_SCHEMA,
                "source_binding_sha256": decode_binding["source_binding_sha256"],
                "execution_binding_sha256": binding_sha,
                "frame_start": stage_start,
                "frame_end": stage_end,
                "target": target_fact,
                "streams": stream_facts,
                "artifacts": artifacts,
                "elapsed_seconds": time.perf_counter() - started,
            }
            atomic_json(current_root / "RECEIPT.json", receipt)
            receipts.append(receipt)
            print(
                json.dumps(
                    {
                        "family": "oe1",
                        "phase": "candidate_decode",
                        "frame_end": stage_end,
                        "windows": selected_windows,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if len(receipts) != N // STAGE_FRAMES:
        raise Bhw2Error("OE1 decode did not retain all 30 stage checkpoints")
    rows: dict[str, dict[str, Any]] = {}
    for window in selected_windows:
        name = oe1.label(window)
        decoded_fact = assemble_oe1_decoded(root, name)
        identity = decoded_fact["bytes"] == target_fact["bytes"] and decoded_fact["sha256"] == target_fact["sha256"]
        if not identity:
            raise Bhw2Error(f"OE1 assembled candidate decode differs: {name}")
        rows[name] = {
            "decoded_tokens": decoded_fact,
            "identity": True,
            "decoder_bit_position": int(decoders[window].bit_position),
        }
    build_path = root / "work/RC64_BUILD.json"
    result = {
        "schema": "ddm_bhw2_oe1_decode_result.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "decode_binding_sha256": binding_sha,
        "target": target_fact,
        "streams": stream_facts,
        "rows": rows,
        "build": {
            **json.loads(build_path.read_text()),
            "receipt": file_fact(build_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(result_path, result)
    return result


def verify_oe1_archive_stream(*, archive: Path, member: Path, stream: Path) -> dict[str, Any]:
    member_payload = member.read_bytes()
    stream_payload = stream.read_bytes()
    packed_member = jg2.read_archive_member(archive)
    if packed_member != member_payload:
        raise Bhw2Error(f"OE1 archive member differs after packing: {archive}")
    purge_runtime_modules()
    residual, _renderer, _renderer_dir = jg2.load_runtime(oe1.RUNTIME)
    parts = residual.read_residual_archive(archive)
    if parts.token_stream != stream_payload:
        raise Bhw2Error(f"OE1 parser extracted a different token stream: {archive}")
    return {
        "archive": file_fact(archive),
        "member": file_fact(member),
        "stream": file_fact(stream),
        "packed_member_identity": True,
        "parser_token_stream_identity": True,
    }


def byte_row(
    *, row: str, screen: dict[str, Any], base_archive: dict[str, Any], candidate_archive: dict[str, Any]
) -> dict[str, Any]:
    edits = int(screen["counts"]["B"])
    delta = int(candidate_archive["bytes"]) - int(base_archive["bytes"])
    bits_per_edit = (8.0 * delta / edits) if edits else None
    admitted = edits > 0 and delta < 0
    return {
        "schema": "ddm_bhw2_byte_row.v1",
        "family": row.split("_")[0],
        "row": row,
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "rows_screened": screen["rows_screened"],
        "selection_rule": screen["selection_rule"],
        "B": screen["counts"]["B"],
        "H": screen["counts"]["H"],
        "W": screen["counts"]["W"],
        "disagreements": screen["disagreements"],
        "B_share_of_disagreements": screen["B_share_of_disagreements"],
        "base_archive": base_archive,
        "candidate_archive": candidate_archive,
        "marginal_bytes_vs_own_base": delta,
        "real_bits_per_edit": bits_per_edit,
        "rate_only_delta_s": delta * S_PER_BYTE,
        "d_seg": "UNMEASURED",
        "d_pose": "UNMEASURED",
        "verdict": "BYTE-ADMITTED-FIRE-MAIN" if admitted else "BYTE-REFUSED",
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    payload = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows)
    return atomic_bytes(path, payload)


def fire_order(family: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    admitted = [row for row in rows if row["verdict"] == "BYTE-ADMITTED-FIRE-MAIN"]
    if not admitted:
        raise Bhw2Error(f"cannot emit a {family} fire order without an admitted byte row")
    result = {
        "schema": "ddm_bhw2_main_fire_order.v1",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "family": family,
        "owner": "MAIN / exclusive n600 scorer-lane custodian",
        "consumer_store": str(STORE / family / "scorer_n600"),
        "fire_trigger": (
            "MAIN explicitly owns the sole idle full-n600 scorer lane; revalidate the exact "
            "base/candidate archive hashes and storage; run receiver plus frozen SegNet/PoseNet "
            "in chunks <=120; retain components; recompute S from components"
        ),
        "admitted_rows": admitted,
        "scorer_ran_here": False,
        "seal_created_here": False,
    }
    atomic_json(STORE / family / "MAIN_FIRE_ORDER.json", result)
    return result


def expected_jf2_stream() -> dict[str, Any]:
    result = json.loads(JF2_RESULT.read_text())
    expected = dict(result["refit_stream"])
    path = JF2_ROOT / "work/tail_refit_e0060_k060000.bin"
    observed = file_fact(path)
    if observed["bytes"] != expected["bytes"] or observed["sha256"] != expected["sha256"]:
        raise Bhw2Error(f"JF2 retained stream mirror drifted: {observed} vs {expected}")
    return observed


def run_jf2() -> dict[str, Any]:
    preflight = storage_preflight("rank1_jf2", RANK1_PROJECTED_BYTES)
    binding = source_binding()
    atomic_json(STORE / "SOURCE_BINDING.json", binding)
    base_replay = run_replay(
        family="jf2",
        root=STORE / "jf2/base_argmax_replay",
        target_path=JF2_TOKENS,
        binding=binding,
        retain_argmax=True,
        expected_streams={"jf2_k060000": expected_jf2_stream()},
    )
    argmax = Path(base_replay["coding_argmax"]["jf2_k060000"]["path"])
    screen = classify_and_materialize(
        row="jf2_k060000",
        root=STORE / "jf2/screen/k060000",
        tokens_path=JF2_TOKENS,
        argmax_path=argmax,
        binding=binding,
    )
    candidate_replay = run_replay(
        family="jf2",
        root=STORE / "jf2/benefit_reencode",
        target_path=Path(screen["payloads"]["benefit_field"]["path"]),
        binding=binding,
        retain_argmax=False,
    )
    candidate_archive = pack_stream_on_archive(
        JF2_ARCHIVE,
        Path(candidate_replay["streams"]["jf2_k060000"]["path"]),
        STORE / "jf2/retained/k060000_benefit_archive.zip",
    )
    library = Path(candidate_replay["build"]["library"]["path"])
    decode = verify_jf2_decode(
        archive=Path(candidate_archive["path"]),
        expected_tokens=Path(screen["payloads"]["benefit_field"]["path"]),
        library=library,
        destination=STORE / "jf2/retained/k060000_benefit_decoded.u8",
    )
    row = byte_row(
        row="jf2_k060000",
        screen=screen,
        base_archive=file_fact(JF2_ARCHIVE),
        candidate_archive=candidate_archive,
    )
    row["receiver_decode"] = decode
    rows = [row]
    fire = fire_order("jf2", rows) if row["verdict"] == "BYTE-ADMITTED-FIRE-MAIN" else None
    result = {
        "schema": "ddm_bhw2_jf2_result.v1",
        "complete": True,
        "terminal": True,
        "rank": 1,
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "preflight": preflight,
        "producer_adaptation": (
            "new exact JG2 receiver-trajectory adapter persists per-stage and assembled "
            "model-matched final coding argmax plus full corrector/encoder checkpoints"
        ),
        "base_replay": base_replay,
        "screen": screen,
        "benefit_reencode": candidate_replay,
        "rows": rows,
        "family_verdict": row["verdict"],
        "main_fire_order": fire,
        "oe1_gate": ("STOP_RANK1_BYTE_ADMITTED" if fire else "OPEN_RANK1_TERMINAL_BYTE_REFUSED"),
    }
    atomic_json(STORE / "jf2/JF2_RESULT.json", result)
    write_jsonl(STORE / "jf2/JF2_ROWS.jsonl", rows)
    return result


def oe1_source_streams() -> dict[str, dict[str, Any]]:
    return {
        oe1.label(window): file_fact(OE1_ROOT / "retained/rungs" / oe1.label(window) / "tokens.rc64")
        for window in oe1.WINDOWS
    }


def oe1_base_archives() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for window in oe1.WINDOWS:
        name = oe1.label(window)
        member = (OE1_ROOT / "retained/rungs" / name / "member.bin").read_bytes()
        path = STORE / "oe1/retained/base_archives" / f"{name}.archive.zip"
        jg2.pack_archive(member, path)
        output[name] = file_fact(path)
    return output


def run_oe1() -> dict[str, Any]:
    jf2_path = STORE / "jf2/JF2_RESULT.json"
    if not jf2_path.is_file():
        raise Bhw2Error("OE1 rank 2 refuses before JF2 rank 1 is terminal")
    jf2 = json.loads(jf2_path.read_text())
    if not jf2.get("terminal"):
        raise Bhw2Error("OE1 rank 2 refuses a nonterminal JF2 receipt")
    if jf2.get("main_fire_order") is not None:
        raise Bhw2Error("OE1 rank 2 STOP: JF2 admitted and already emitted MAIN fire order")
    preflight = storage_preflight("rank2_oe1", RANK2_PROJECTED_BYTES)
    binding = source_binding()
    base_replay = run_replay(
        family="oe1",
        root=STORE / "oe1/base_argmax_replay",
        target_path=oe1.TOKENS,
        binding=binding,
        retain_argmax=True,
        windows=oe1.WINDOWS,
        expected_streams=oe1_source_streams(),
    )
    screens: dict[str, dict[str, Any]] = {}
    for name, fact in base_replay["coding_argmax"].items():
        screens[name] = classify_and_materialize(
            row=f"oe1_{name}",
            root=STORE / "oe1/screen" / name,
            tokens_path=oe1.TOKENS,
            argmax_path=Path(fact["path"]),
            binding=binding,
        )

    groups: dict[str, list[int]] = {}
    benefit_facts: dict[str, dict[str, Any]] = {}
    for window in oe1.WINDOWS:
        name = oe1.label(window)
        fact = screens[name]["payloads"]["benefit_field"]
        benefit_facts[name] = fact
        groups.setdefault(str(fact["sha256"]), []).append(window)
    candidate_streams: dict[str, dict[str, Any]] = {}
    replays: list[dict[str, Any]] = []
    decodes: list[dict[str, Any]] = []
    decoded_rows: dict[str, dict[str, Any]] = {}
    for group_index, windows in enumerate(groups.values()):
        first = oe1.label(windows[0])
        replay = run_replay(
            family="oe1",
            root=STORE / "oe1/benefit_reencode" / f"group_{group_index:02d}",
            target_path=Path(benefit_facts[first]["path"]),
            binding=binding,
            retain_argmax=False,
            windows=windows,
        )
        replays.append(replay)
        candidate_streams.update(replay["streams"])
        group_streams = {oe1.label(window): replay["streams"][oe1.label(window)] for window in windows}
        decode = decode_oe1_streams(
            root=STORE / "oe1/benefit_decode" / f"group_{group_index:02d}",
            target_path=Path(benefit_facts[first]["path"]),
            streams=group_streams,
            windows=windows,
            binding=binding,
        )
        decodes.append(decode)
        decoded_rows.update(decode["rows"])

    base_archives = oe1_base_archives()
    member_prefix = oe1.member_prefix(STORE / "oe1/member_prefix")[0]
    rows: list[dict[str, Any]] = []
    for window in oe1.WINDOWS:
        name = oe1.label(window)
        member = member_prefix + Path(candidate_streams[name]["path"]).read_bytes()
        member_path = STORE / "oe1/retained/benefit_members" / f"{name}.member.bin"
        atomic_bytes(member_path, member)
        candidate_path = STORE / "oe1/retained/benefit_archives" / f"{name}.archive.zip"
        jg2.pack_archive(member, candidate_path)
        archive_parseback = verify_oe1_archive_stream(
            archive=candidate_path,
            member=member_path,
            stream=Path(candidate_streams[name]["path"]),
        )
        row = byte_row(
            row=f"oe1_{name}",
            screen=screens[name],
            base_archive=base_archives[name],
            candidate_archive=file_fact(candidate_path),
        )
        row["receiver_decode"] = decoded_rows[name]
        row["archive_parseback"] = archive_parseback
        rows.append(row)
    fire = fire_order("oe1", rows) if any(row["verdict"] == "BYTE-ADMITTED-FIRE-MAIN" for row in rows) else None
    argmax_hashes = {name: fact["sha256"] for name, fact in base_replay["coding_argmax"].items()}
    result = {
        "schema": "ddm_bhw2_oe1_result.v1",
        "complete": True,
        "terminal": True,
        "rank": 2,
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "preflight": preflight,
        "producer_adaptation": (
            "exact OE1 producer/checkpoint adapter now persists five per-stage and assembled "
            "online-mixture final coding argmax payloads alongside all five encoder states"
        ),
        "base_replay": base_replay,
        "coding_argmax_hashes": argmax_hashes,
        "all_five_argmax_byte_identical": len(set(argmax_hashes.values())) == 1,
        "screens": screens,
        "benefit_reencodes": replays,
        "benefit_decodes": decodes,
        "rows": rows,
        "family_verdict": ("BYTE-ADMITTED-FIRE-MAIN" if fire else "BYTE-REFUSED"),
        "main_fire_order": fire,
    }
    atomic_json(STORE / "oe1/OE1_RESULT.json", result)
    write_jsonl(STORE / "oe1/OE1_ROWS.jsonl", rows)
    return result


def write_manifest() -> dict[str, Any]:
    artifacts = [
        file_fact(path) for path in sorted(STORE.rglob("*")) if path.is_file() and path.name != "MANIFEST.json"
    ]
    result = {
        "schema": "ddm_bhw2_manifest.v1",
        "root": str(STORE),
        "artifact_count": len(artifacts),
        "artifact_bytes": sum(int(fact["bytes"]) for fact in artifacts),
        "artifacts": artifacts,
        "retention": (
            "No listed payload may be moved or deleted without replacement custody; "
            "all stage checkpoints are distinct and preserved."
        ),
    }
    atomic_json(STORE / "MANIFEST.json", result)
    return result


def self_test() -> None:
    base = np.asarray(
        [[0.7, 0.1, 0.1, 0.05, 0.05], [0.2, 0.2, 0.3, 0.1, 0.2]],
        dtype=np.float32,
    )
    contracted = base + np.float32(0.5) * (oe1.UNIFORM - base)
    if not np.array_equal(base.argmax(axis=1), contracted.argmax(axis=1)):
        raise Bhw2Error("positive uniform contraction changed an untied ordering")
    labels = fcd1.classify_pool(
        np.asarray([0, 1, 2, 3], dtype=np.uint8),
        np.asarray([1, 2, 3, 3], dtype=np.uint8),
        np.asarray([1, 1, 4, 3], dtype=np.uint8),
    )
    if tuple(int(labels[key].sum()) for key in ("benefit", "harm", "wash")) != (
        1,
        1,
        1,
    ):
        raise Bhw2Error("B/H/W canonical classifier control failed")
    print(json.dumps({"self_test": "PASS", "positions": POSITIONS}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "jf2", "oe1", "manifest", "self-test"))
    args = parser.parse_args()
    if args.stage == "self-test":
        self_test()
        return
    if args.stage == "preflight":
        result = {
            "source_binding": source_binding(),
            "rank1": storage_preflight("rank1_jf2", RANK1_PROJECTED_BYTES),
            "rank2_argmax_payload_floor_bytes": RANK2_ARGMAX_FLOOR_BYTES,
        }
    elif args.stage == "jf2":
        result = run_jf2()
    elif args.stage == "oe1":
        result = run_oe1()
    else:
        result = write_manifest()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
