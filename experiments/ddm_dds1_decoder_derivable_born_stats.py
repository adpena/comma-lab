#!/usr/bin/env python3
"""DDS1: measure whether GF1 born statistics are derivable from decoder state.

This is a scorer-free SCREEN.  It consumes the retained AFR1 exact token field,
the retained DF1 coding argmax/pmax fields, and the retained GF1 packet/field.
It never touches the live RXC1 instrument or its store and never claims physical
coder bytes from conditional-entropy estimates.

Stages are restartable and retain every materialized scientific payload:

``materialize``
    Select 120 random pairs by seed, snapshot the counted AFR1 archive sections,
    and persist the exact GF1 feature plus five causal zero-byte surrogates.
``analyze``
    Cross-fit nested log-odds maps by pair, measure wrong-branch overlap, and
    write the typed derive-or-close receipt.
``manifest``
    Hash every retained output except the self-referential manifest.

The five formulations covering the charter's four routes are:

* ``prefix``: class of the latest already-decoded causal neighbour plus L1
  distance to an inter-class edge that was already observable before the site.
* ``refit``: the exact HG1 fitter applied to the complete temporal prefix, then
  the last fitted parameters extrapolated one pair.
* ``model``: current HPAC argmax plus causal boundary distance over argmaxes
  born in earlier groups.
* ``previous``: previous decoded class and its shipped boundary bucket.
* ``temporal_mode``: per-site modal class over the fully decoded frame prefix.

All current-frame prefix features honor the shipped 64x64, delta=2 group order.
Positions in the current group are never visible to one another.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Final

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
for root in (REPO, REPO / "src", REPO / "experiments"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments import ddm_hg1_heterogeneous_analytic_generator_gate as hg1
from tac.boundary_math.analytic_lane_render_band import (
    LaneBandRenderConfig,
    build_lane_band_pairs_from_lstars,
    rasterize_lane_coverage_range_dependent,
)

SCHEMA: Final = "ddm_dds1_decoder_derivable_born_stats.v2"
AXIS: Final = "[macOS-CPU scorer-free conditional-codelength SCREEN]"
N, HEIGHT, WIDTH = 600, 384, 512
PLANE = HEIGHT * WIDTH
SAMPLE_N = 120
DEFAULT_SEED = 20_260_901
PATCH = 64
DELTA = 2
GROUPS = (1 + DELTA) * PATCH - DELTA
MAX_DISTANCE = 4
CLASSES = 5
CONTEXT_CELLS = CLASSES * CLASSES * (MAX_DISTANCE + 1) * 2

STORE = Path("/Volumes/VertigoDataTier/pact/ddm_dds1/measurement_v2")
X_PATH = Path(
    "/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/identity_v1/"
    "out/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
ARCHIVE_PATH = Path(
    "/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/"
    "runtime_candidate_native/archive.zip"
)
GF1_PACKET = Path(
    "/Volumes/APDataStore/pact/ddm_gf1_generator_form_on_lb1_field/retained/"
    "gf1_lb1_fit.packet"
)
GF1_FIELD = Path(
    "/Volumes/APDataStore/pact/ddm_gf1_generator_form_on_lb1_field/retained/"
    "generated_from_lb1_fit.u8"
)
DF1_ROOT = Path(
    "/Volumes/APDataStore/pact/ddm_df1_dddb_field/measurement_v1/retained/fields"
)
ARGMAX_PATH = DF1_ROOT / "position_coding_argmax.u8.bin"
PMAX_PATH = DF1_ROOT / "position_coding_pmax.f32le.bin"

EXPECTED: Final = {
    "x": (117_964_800, "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"),
    "archive": (180_002, "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25"),
    "gf1_packet": (47_603, "87d79345982dde33e30ca328de2dcde9c66c20e12e7729a3690ae8e23b4e1497"),
    "gf1_field": (117_964_800, "4026c4e2c805beb5b79be2879bb4a84311655d0d7d80dbc766654847522a5d19"),
    "argmax": (117_964_800, "db498280c22c3aa1b787310e25435116911933216cae558f309f8b10baf7994e"),
    "pmax": (471_859_200, "f37e3d8a21d02647437bf950d7a8a75b751c2a9644c7b8ad48aca2833be4794b"),
}

SURROGATE_NAMES: Final = ("prefix", "refit", "model", "previous", "temporal_mode")
FIELD_DTYPES: Final = {
    "x": np.uint8,
    "argmax": np.uint8,
    "pmax": np.dtype("<f4"),
    "gf1_class": np.uint8,
    "gf1_boundary_d": np.uint8,
    "prefix_class": np.uint8,
    "prefix_boundary_d": np.uint8,
    "refit_class": np.uint8,
    "refit_boundary_d": np.uint8,
    "model_class": np.uint8,
    "model_boundary_d": np.uint8,
    "previous_class": np.uint8,
    "previous_boundary_d": np.uint8,
    "temporal_mode_class": np.uint8,
    "temporal_mode_boundary_d": np.uint8,
}


class Dds1Error(RuntimeError):
    """A custody, causality, checkpoint, or analysis invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_source(path: Path, key: str) -> dict[str, Any]:
    expected_bytes, expected_sha = EXPECTED[key]
    fact = file_fact(path)
    if fact["bytes"] != expected_bytes or fact["sha256"] != expected_sha:
        raise Dds1Error(f"{key} source drift: {fact}")
    return fact


def selected_pairs(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(N, size=SAMPLE_N, replace=False)).astype(np.int16)


def boundary_buckets(field: np.ndarray) -> np.ndarray:
    """Exact twin of residual_archive._boundary_buckets, clipped to [0,4]."""

    a = np.asarray(field)
    if a.shape != (HEIGHT, WIDTH):
        raise Dds1Error(f"boundary field shape changed: {a.shape}")
    edge = np.zeros(a.shape, dtype=bool)
    edge[1:] |= a[1:] != a[:-1]
    edge[:-1] |= a[:-1] != a[1:]
    edge[:, 1:] |= a[:, 1:] != a[:, :-1]
    edge[:, :-1] |= a[:, :-1] != a[:, 1:]
    result = np.full(a.shape, MAX_DISTANCE, dtype=np.uint8)
    active = edge.copy()
    result[active] = 0
    for distance in range(1, MAX_DISTANCE):
        grown = active.copy()
        grown[1:] |= active[:-1]
        grown[:-1] |= active[1:]
        grown[:, 1:] |= active[:, :-1]
        grown[:, :-1] |= active[:, 1:]
        active = grown
        result[(result == MAX_DISTANCE) & active] = distance
    return result


def _shift_int(source: np.ndarray, dy: int, dx: int, fill: int) -> np.ndarray:
    output = np.full(source.shape, fill, dtype=source.dtype)
    y_src = slice(max(0, -dy), min(HEIGHT, HEIGHT - dy))
    x_src = slice(max(0, -dx), min(WIDTH, WIDTH - dx))
    y_dst = slice(max(0, dy), min(HEIGHT, HEIGHT + dy))
    x_dst = slice(max(0, dx), min(WIDTH, WIDTH + dx))
    output[y_dst, x_dst] = source[y_src, x_src]
    return output


def prefix_features(current: np.ndarray, previous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return class/boundary features visible strictly before each decode group.

    ``prefix_class`` selects the most recent of L/U/UR/UL whose local group is
    strictly smaller than the current site's group; if none exists it falls
    back to the fully decoded previous frame. ``prefix_boundary_d`` is the L1
    distance (clipped at 4) to an inter-class edge whose two endpoint tokens
    were both decoded in an earlier group.  The implementation computes the
    eventual discovery time offline, but the ``discover < current_group`` test
    is exactly the state a streaming receiver has at that site.
    """

    yy, xx = np.indices((HEIGHT, WIDTH), dtype=np.int16)
    group = ((xx % PATCH) + DELTA * (yy % PATCH)).astype(np.int16)

    best_group = np.full((HEIGHT, WIDTH), -1, dtype=np.int16)
    prefix_class = np.asarray(previous, dtype=np.uint8).copy()
    # Fixed priority breaks equal-group ties without observing the current group.
    for dy, dx in ((0, -1), (-1, 0), (-1, 1), (-1, -1)):
        neighbour_group = _shift_int(group, -dy, -dx, GROUPS).astype(np.int16)
        neighbour_class = _shift_int(np.asarray(current, dtype=np.uint8), -dy, -dx, 0)
        legal = (neighbour_group < group) & (neighbour_group > best_group)
        prefix_class[legal] = neighbour_class[legal]
        best_group[legal] = neighbour_group[legal]

    # Earliest group at which each true current-frame edge endpoint is known.
    discover = np.full((HEIGHT, WIDTH), GROUPS + 1, dtype=np.int16)
    for dy, dx in ((1, 0), (0, 1)):
        neighbour_group = _shift_int(group, -dy, -dx, GROUPS + 1).astype(np.int16)
        neighbour_class = _shift_int(np.asarray(current, dtype=np.uint8), -dy, -dx, 255)
        valid = neighbour_class != 255
        differs = valid & (np.asarray(current) != neighbour_class)
        # The edge becomes receiver-visible immediately after the later endpoint's
        # group is decoded.  A site in group ``g`` may therefore use an edge whose
        # later endpoint has group < g, but never an edge born in its own group.
        when = np.maximum(group, neighbour_group)
        discover[differs] = np.minimum(discover[differs], when[differs])
        back = _shift_int(np.where(differs, when, GROUPS + 1).astype(np.int16), dy, dx, GROUPS + 1)
        discover = np.minimum(discover, back)

    causal_d = np.full((HEIGHT, WIDTH), MAX_DISTANCE, dtype=np.uint8)
    reachable = discover.copy()
    for distance in range(MAX_DISTANCE):
        visible = reachable < group
        causal_d[(causal_d == MAX_DISTANCE) & visible] = distance
        if distance + 1 < MAX_DISTANCE:
            reachable = np.minimum.reduce(
                (
                    reachable,
                    _shift_int(reachable, 1, 0, GROUPS + 1),
                    _shift_int(reachable, -1, 0, GROUPS + 1),
                    _shift_int(reachable, 0, 1, GROUPS + 1),
                    _shift_int(reachable, 0, -1, GROUPS + 1),
                )
            )
    return prefix_class, causal_d


def causal_refit(
    history: np.ndarray,
    prefix_counts: np.ndarray,
    prefix_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit HG1 on the complete temporal prefix and extrapolate its last parameters.

    Horizon and movable fitting are pair-local, lane fitting tracks slots over
    the supplied sequence, and MyCar fitting is a temporal majority.  Passing
    exactly ``X[:pair]`` therefore runs the real generic fitter on all complete
    frames available to the receiver.  Reusing the final fitted parameters for
    the next pair is the causal extrapolation under test.
    """

    if prefix_frames == 0:
        return np.zeros((HEIGHT, WIDTH), dtype=np.uint8), {
            "cold_start": True,
            "lane_lines": 0,
            "movable_boxes": 0,
            "mycar_pixels": 0,
        }

    if len(history) != prefix_frames:
        raise Dds1Error("temporal prefix length drifted")
    frame = np.asarray(history[-1], dtype=np.uint8)
    road = frame == hg1.CLASS_ROAD
    first = np.argmax(road, axis=0)
    first[~np.any(road, axis=0)] = HEIGHT
    first = ndimage.median_filter(first.astype(np.int32), size=17, mode="nearest")
    knots = np.clip(first[hg1.HORIZON_X], 0, HEIGHT)
    horizon = np.rint(
        np.interp(np.arange(WIDTH, dtype=np.float64), hg1.HORIZON_X, knots)
    ).astype(np.int32)
    y_grid = np.arange(HEIGHT, dtype=np.int32)[:, None]
    output = np.where(
        y_grid >= horizon[None, :], hg1.CLASS_ROAD, hg1.CLASS_UNDRIVABLE
    ).astype(np.uint8)

    cfg = LaneBandRenderConfig()
    lines, _stats = build_lane_band_pairs_from_lstars(history, cfg)
    coverage = rasterize_lane_coverage_range_dependent(
        lines[-1],
        h=HEIGHT,
        w=WIDTH,
        softness=cfg.softness,
        dash_gate=cfg.dash_gate,
        dash_forward_max_m=cfg.dash_forward_max_m,
        v_h=cfg.v_h,
        cx=cfg.cx,
    )
    output[coverage >= 0.5] = hg1.CLASS_LANE

    boxes = hg1.component_boxes(frame)
    for y0, x0, y1, x1 in boxes:
        output[y0:y1, x0:x1] = hg1.CLASS_MOVABLE

    mycar = prefix_counts[hg1.CLASS_MYCAR] * 2 >= prefix_frames
    output[mycar] = hg1.CLASS_MYCAR
    return output, {
        "cold_start": False,
        "lane_lines": len(lines[-1]),
        "movable_boxes": len(boxes),
        "mycar_pixels": int(np.count_nonzero(mycar)),
        "horizon_knots": knots.astype(int).tolist(),
    }


def _paths(root: Path) -> dict[str, Path]:
    fields = root / "retained" / "surrogate_fields"
    return {name: fields / f"{name}.bin" for name in FIELD_DTYPES}


def _partial_paths(root: Path) -> dict[str, Path]:
    return {name: path.with_suffix(path.suffix + ".partial") for name, path in _paths(root).items()}


def _open_output_fields(root: Path, mode: str, *, partial: bool) -> dict[str, np.memmap]:
    paths = _partial_paths(root) if partial else _paths(root)
    return {
        name: np.memmap(paths[name], dtype=dtype, mode=mode, shape=(SAMPLE_N, HEIGHT, WIDTH))
        for name, dtype in FIELD_DTYPES.items()
    }


def snapshot_archive_sections(root: Path) -> dict[str, Any]:
    destination = root / "retained" / "counted_decoder_state"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        outer = archive.read("p")
    if len(outer) < 14:
        raise Dds1Error("AFR1 archive member is truncated")
    magic, version, codec, table_mode, reserved, hpac_n, semantic_n, carrier_n = struct.unpack_from(
        "<4sBBBBHHH", outer, 0
    )
    if magic != b"RX1M" or version != 1 or hpac_n != 13_515:
        raise Dds1Error("AFR1 RX1 model header drifted")
    cursor = 14
    sections: dict[str, bytes] = {}
    for name, size in (
        ("hpac_model_counted", hpac_n),
        ("semantic_counted", semantic_n),
        ("carrier_dxi_counted", carrier_n),
    ):
        sections[name] = outer[cursor : cursor + size]
        cursor += size
    # The compact fixed table is 96 B; the rest is the exact RC64 stream.
    sections["mixer_table_counted"] = outer[cursor : cursor + 96]
    sections["token_stream_counted"] = outer[cursor + 96 :]
    facts = {}
    for name, payload in sections.items():
        path = destination / f"{name}.bin"
        if path.exists() and path.read_bytes() != payload:
            raise Dds1Error(f"pre-existing counted-state snapshot drifted: {path}")
        if not path.exists():
            atomic_bytes(path, payload)
        facts[name] = file_fact(path)
    facts["header"] = {
        "magic": magic.decode("ascii"),
        "version": version,
        "codec": codec,
        "table_mode": table_mode,
        "reserved": reserved,
        "hpac_bytes": hpac_n,
        "semantic_bytes": semantic_n,
        "carrier_bytes": carrier_n,
    }
    return facts


def run_materialize(root: Path, seed: int) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    paths = _paths(root)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    source_facts = {
        "x": verify_source(X_PATH, "x"),
        "archive": verify_source(ARCHIVE_PATH, "archive"),
        "gf1_packet": verify_source(GF1_PACKET, "gf1_packet"),
        "gf1_field": verify_source(GF1_FIELD, "gf1_field"),
        "argmax": verify_source(ARGMAX_PATH, "argmax"),
        "pmax": verify_source(PMAX_PATH, "pmax"),
    }
    pairs = selected_pairs(seed)
    config = {
        "schema": SCHEMA,
        "stage": "materialize",
        "axis": AXIS,
        "score_claim": False,
        "scope": "SCREEN seeded random n=120 pairs; never a prefix; no physical-byte claim",
        "seed": seed,
        "selection": {"mode": "seeded_random_without_replacement", "pairs": pairs.tolist()},
        "sources": source_facts,
        "store": str(root),
        "git_head_before_serializer": git_head(),
        "runner": file_fact(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "decoder_order": {"patch": PATCH, "delta": DELTA, "groups": GROUPS},
    }
    atomic_json(root / "RUN_CONFIG.json", config)
    counted_state = snapshot_archive_sections(root)

    completion = root / "stage_01_materialize_complete.json"
    if completion.is_file() and all(path.is_file() for path in paths.values()):
        prior = json.loads(completion.read_text())
        for name, path in paths.items():
            if prior["fields"][name]["sha256"] != sha256_file(path):
                raise Dds1Error(f"completed materialization field drifted: {name}")
        return prior

    partial = _partial_paths(root)
    checkpoint_path = root / "checkpoint_materialize_latest.json"
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("seed") != seed or checkpoint.get("pairs") != pairs.tolist():
            raise Dds1Error("materialize checkpoint binding drifted")
        next_sample = int(checkpoint["next_sample"])
        if not all(path.is_file() for path in partial.values()):
            raise Dds1Error("materialize checkpoint exists without every partial field")
        outputs = _open_output_fields(root, "r+", partial=True)
    else:
        if any(path.exists() for path in partial.values()):
            raise Dds1Error("partial fields exist without a binding checkpoint")
        outputs = _open_output_fields(root, "w+", partial=True)
        next_sample = 0

    x = np.memmap(X_PATH, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    argmax = np.memmap(ARGMAX_PATH, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    pmax = np.memmap(PMAX_PATH, dtype="<f4", mode="r", shape=(N, HEIGHT, WIDTH))
    gf1 = np.memmap(GF1_FIELD, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))

    prefix_counts = np.zeros((CLASSES, HEIGHT, WIDTH), dtype=np.uint16)
    advanced_to = 0
    refit_rows: list[dict[str, Any]] = []
    if next_sample:
        prior_diag = root / "REFIT_DIAGNOSTICS.partial.json"
        if not prior_diag.is_file():
            raise Dds1Error("materialize checkpoint lacks refit diagnostics")
        refit_rows = json.loads(prior_diag.read_text())["rows"]

    for sample_index, pair in enumerate(pairs.tolist()):
        while advanced_to < pair:
            frame = np.asarray(x[advanced_to])
            for cls in range(CLASSES):
                prefix_counts[cls] += frame == cls
            advanced_to += 1
        if sample_index < next_sample:
            continue

        current = np.asarray(x[pair], dtype=np.uint8)
        previous = np.asarray(x[pair - 1], dtype=np.uint8) if pair else np.zeros_like(current)
        generated = np.asarray(gf1[pair], dtype=np.uint8)
        mode_class = np.argmax(prefix_counts, axis=0).astype(np.uint8)
        prefix_class, prefix_d = prefix_features(current, previous)
        refit_class, diagnostics = causal_refit(x[:pair], prefix_counts, pair)
        current_argmax = np.asarray(argmax[pair], dtype=np.uint8)
        previous_argmax = (
            np.asarray(argmax[pair - 1], dtype=np.uint8) if pair else np.zeros_like(current)
        )
        _model_anchor, model_d = prefix_features(current_argmax, previous_argmax)

        values = {
            "x": current,
            "argmax": current_argmax,
            "pmax": np.asarray(pmax[pair], dtype="<f4"),
            "gf1_class": generated,
            "gf1_boundary_d": boundary_buckets(generated),
            "prefix_class": prefix_class,
            "prefix_boundary_d": prefix_d,
            "refit_class": refit_class,
            "refit_boundary_d": boundary_buckets(refit_class),
            "model_class": current_argmax,
            "model_boundary_d": model_d,
            "previous_class": previous,
            "previous_boundary_d": (
                boundary_buckets(previous) if pair else np.full_like(previous, 4)
            ),
            "temporal_mode_class": mode_class,
            "temporal_mode_boundary_d": boundary_buckets(mode_class),
        }
        for name, value in values.items():
            outputs[name][sample_index] = value
            outputs[name].flush()

        diagnostics.update(
            {
                "pair": pair,
                "sample_index": sample_index,
                "gf1_class_agreement": float(np.mean(refit_class == generated)),
                "gf1_boundary_agreement": float(
                    np.mean(values["refit_boundary_d"] == values["gf1_boundary_d"])
                ),
            }
        )
        refit_rows.append(diagnostics)
        atomic_json(root / "REFIT_DIAGNOSTICS.partial.json", {"rows": refit_rows})
        atomic_json(
            checkpoint_path,
            {
                "schema": SCHEMA,
                "stage": "materialize",
                "seed": seed,
                "pairs": pairs.tolist(),
                "next_sample": sample_index + 1,
                "partial_fields": {name: str(path) for name, path in partial.items()},
            },
        )

    for output in outputs.values():
        output.flush()
    del outputs, x, argmax, pmax, gf1
    for name, path in partial.items():
        os.replace(path, paths[name])

    diagnostics_path = root / "REFIT_DIAGNOSTICS.json"
    os.replace(root / "REFIT_DIAGNOSTICS.partial.json", diagnostics_path)
    receipt = {
        "schema": SCHEMA,
        "stage": "materialize",
        "axis": AXIS,
        "score_claim": False,
        "seed": seed,
        "pairs": pairs.tolist(),
        "sample_pairs": SAMPLE_N,
        "sample_sites": SAMPLE_N * PLANE,
        "counted_decoder_state": counted_state,
        "fields": {name: file_fact(path) for name, path in paths.items()},
        "refit_diagnostics": file_fact(diagnostics_path),
        "causality": {
            "prefix": "only group<current_group current-frame tokens, else previous frame",
            "refit": "exact HG1 fit over complete prior frames, then last-parameter extrapolation",
            "model": "current HPAC argmax plus boundaries born strictly before the current group",
            "previous": "previous full decoded frame and its shipped boundary bucket",
            "temporal_mode": "per-site mode over frames strictly before current",
            "gf1": "full generated field; legal only when the counted GF1 packet is transmitted",
        },
    }
    atomic_json(completion, receipt)
    return receipt


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    output = np.empty_like(value, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def _bits(logit: np.ndarray, flip: np.ndarray, offset: np.ndarray) -> tuple[float, float]:
    u = logit + offset
    per_site = np.where(
        flip,
        np.logaddexp(0.0, -u),
        np.logaddexp(0.0, u),
    ) / np.log(2.0)
    return float(per_site.sum()), float(per_site[flip].sum())


def fit_crossfold(
    logit: np.ndarray,
    flip: np.ndarray,
    cell: np.ndarray,
    fold: np.ndarray,
    n_cells: int,
) -> dict[str, Any]:
    total_bits = 0.0
    wrong_bits = 0.0
    fold_rows = []
    for test_fold in (0, 1):
        train = fold != test_fold
        test = ~train
        beta = np.zeros(n_cells, dtype=np.float64)
        train_cell = cell[train]
        train_logit = logit[train]
        train_flip = flip[train].astype(np.float64)
        iterations = 0
        max_step = 0.0
        for iteration in range(24):
            q = _sigmoid(train_logit + beta[train_cell])
            gradient = np.bincount(
                train_cell, weights=train_flip - q, minlength=n_cells
            )
            hessian = np.bincount(
                train_cell, weights=q * (1.0 - q), minlength=n_cells
            )
            step = np.divide(
                gradient,
                hessian,
                out=np.zeros_like(gradient),
                where=hessian > 0.0,
            )
            step = np.clip(step, -4.0, 4.0)
            beta += step
            iterations = iteration + 1
            max_step = float(np.max(np.abs(step)))
            if max_step < 1e-10:
                break
        bits_total, bits_wrong = _bits(logit[test], flip[test], beta[cell[test]])
        total_bits += bits_total
        wrong_bits += bits_wrong
        fold_rows.append(
            {
                "test_fold": test_fold,
                "train_sites": int(train.sum()),
                "test_sites": int(test.sum()),
                "iterations": iterations,
                "last_max_step": max_step,
                "active_cells": int(np.count_nonzero(np.bincount(train_cell, minlength=n_cells))),
                "beta_min": float(beta.min()),
                "beta_max": float(beta.max()),
                "total_bits": bits_total,
                "wrong_bits": bits_wrong,
            }
        )
    return {"total_bits": total_bits, "wrong_bits": wrong_bits, "folds": fold_rows}


def feature_id(cls: np.ndarray, distance: np.ndarray, argmax: np.ndarray) -> np.ndarray:
    if np.any(cls >= CLASSES) or np.any(distance > MAX_DISTANCE):
        raise Dds1Error("feature domain escaped")
    return (
        (
            (cls.astype(np.int16) * CLASSES + argmax.astype(np.int16))
            * (MAX_DISTANCE + 1)
            + distance.astype(np.int16)
        )
        * 2
        + (cls == argmax)
    ).astype(np.int16)


def run_analyze(root: Path, seed: int) -> dict[str, Any]:
    materialized = root / "stage_01_materialize_complete.json"
    if not materialized.is_file():
        raise Dds1Error("analyze requires completed materialize stage")
    materialize_receipt = json.loads(materialized.read_text())
    if materialize_receipt["seed"] != seed:
        raise Dds1Error("analysis seed differs from materialized sample")
    paths = _paths(root)
    fields = {
        name: np.memmap(paths[name], dtype=dtype, mode="r", shape=(SAMPLE_N, HEIGHT, WIDTH))
        for name, dtype in FIELD_DTYPES.items()
    }

    rng = np.random.default_rng(seed + 1)
    pair_fold = np.zeros(SAMPLE_N, dtype=np.uint8)
    pair_fold[rng.permutation(SAMPLE_N)[SAMPLE_N // 2 :]] = 1
    pmax_full = np.asarray(fields["pmax"], dtype=np.float64).reshape(-1)
    live = pmax_full < 1.0
    if not np.all((pmax_full[live] > 0.0) & (pmax_full[live] < 1.0)):
        raise Dds1Error("pmax left probability domain")
    q = 1.0 - pmax_full[live]
    logit = np.log(q) - np.log1p(-q)
    x = np.asarray(fields["x"]).reshape(-1)[live]
    argmax = np.asarray(fields["argmax"]).reshape(-1)[live]
    flip = x != argmax
    fold_full = np.repeat(pair_fold, PLANE)[live]
    base_total_bits, base_wrong_bits = _bits(logit, flip, np.zeros_like(logit))

    gf1_cls = np.asarray(fields["gf1_class"]).reshape(-1)[live]
    gf1_d = np.asarray(fields["gf1_boundary_d"]).reshape(-1)[live]
    gf1_feature = feature_id(gf1_cls, gf1_d, argmax)
    gf1_model = fit_crossfold(logit, flip, gf1_feature, fold_full, CONTEXT_CELLS)
    gf1_total_gain = base_total_bits - gf1_model["total_bits"]
    gf1_wrong_gain = base_wrong_bits - gf1_model["wrong_bits"]

    wrong_base_per_site = np.logaddexp(0.0, -logit[flip]) / np.log(2.0)
    routes: dict[str, Any] = {}
    for route in SURROGATE_NAMES:
        route_cls = np.asarray(fields[f"{route}_class"]).reshape(-1)[live]
        route_d = np.asarray(fields[f"{route}_boundary_d"]).reshape(-1)[live]
        route_feature = feature_id(route_cls, route_d, argmax)
        route_model = fit_crossfold(logit, flip, route_feature, fold_full, CONTEXT_CELLS)
        route_total_gain = base_total_bits - route_model["total_bits"]
        route_wrong_gain = base_wrong_bits - route_model["wrong_bits"]
        raw_overlap_total = route_total_gain / gf1_total_gain if gf1_total_gain > 0.0 else None
        raw_overlap_wrong = route_wrong_gain / gf1_wrong_gain if gf1_wrong_gain > 0.0 else None
        overlap_total = (
            float(np.clip(raw_overlap_total, 0.0, 1.0))
            if raw_overlap_total is not None
            else None
        )
        overlap_wrong = (
            float(np.clip(raw_overlap_wrong, 0.0, 1.0))
            if raw_overlap_wrong is not None
            else None
        )
        equality = route_cls == gf1_cls
        wrong_equality = equality[flip]
        tuple_equality = route_feature == gf1_feature
        wrong_tuple_equality = tuple_equality[flip]
        if overlap_wrong is None or overlap_wrong < 0.10:
            route_verdict = "CLOSED"
        elif overlap_wrong < 0.90:
            route_verdict = "PARTIAL"
        else:
            route_verdict = "DERIVED"
        routes[route] = {
            "typed_outcome": route_verdict,
            "class_agreement_all_live": float(np.mean(equality)),
            "class_agreement_wrong_sites": float(np.mean(wrong_equality)),
            "wrong_surprise_weighted_class_agreement": float(
                wrong_base_per_site[wrong_equality].sum() / wrong_base_per_site.sum()
            ),
            "boundary_bucket_agreement_all_live": float(np.mean(route_d == gf1_d)),
            "boundary_bucket_agreement_wrong_sites": float(np.mean(route_d[flip] == gf1_d[flip])),
            "exact_tuple_agreement_all_live": float(np.mean(tuple_equality)),
            "exact_tuple_agreement_wrong_sites": float(np.mean(wrong_tuple_equality)),
            "wrong_surprise_weighted_exact_tuple_agreement": float(
                wrong_base_per_site[wrong_tuple_equality].sum() / wrong_base_per_site.sum()
            ),
            "surrogate_model": route_model,
            "surrogate_total_gain_screen_bits": route_total_gain,
            "surrogate_wrong_gain_screen_bits": route_wrong_gain,
            "raw_predictive_gain_ratio_total": raw_overlap_total,
            "raw_predictive_gain_ratio_wrong": raw_overlap_wrong,
            "predictive_value_overlap_total_fraction": overlap_total,
            "predictive_value_overlap_wrong_fraction": overlap_wrong,
        }

    eligible = {
        name: row["predictive_value_overlap_wrong_fraction"]
        for name, row in routes.items()
        if row["predictive_value_overlap_wrong_fraction"] is not None
    }
    if not eligible:
        verdict = "CLOSED"
        best_route = max(routes, key=lambda name: routes[name]["wrong_surprise_weighted_class_agreement"])
        best_overlap = None
        reason = "GF1 context had no positive cross-fitted wrong-branch value to reproduce"
    else:
        best_route = max(eligible, key=eligible.get)
        best_overlap = float(eligible[best_route])
        if best_overlap < 0.10:
            verdict = "CLOSED"
            reason = "best causal surrogate captured below the pre-registered 10% wrong-half overlap floor"
        elif best_overlap < 0.90:
            verdict = "PARTIAL"
            reason = "a causal surrogate captured at least 10% but did not reproduce GF1 statistics at high fidelity"
        else:
            verdict = "DERIVED"
            reason = "a causal surrogate reproduced at least 90% of GF1 wrong-half value"

    result = {
        "schema": SCHEMA,
        "stage": "analyze",
        "axis": AXIS,
        "score_claim": False,
        "physical_byte_claim": False,
        "scope": {
            "label": "SCREEN",
            "selection": "seeded random n=120 pairs, pair-level 2-fold cross-fit, never a prefix",
            "seed": seed,
            "pairs": materialize_receipt["pairs"],
            "sample_sites": SAMPLE_N * PLANE,
            "live_sites": int(live.sum()),
            "wrong_sites": int(flip.sum()),
            "note": "screen bits are conditional-codelength estimates, not real coder or archive bytes",
        },
        "demand_type": {
            "site": "s=(pair:u16,y:u16,x:u16) in shipped 64x64 delta=2 causal group order",
            "candidate_1": "BornContext(s)=(generated_class:u3, hpac_model_class:u3, boundary_distance:u3, agreement:u1)",
            "candidate_2": "PeelContext(s,k)=(rung:u3, generated_class:u3, hpac_model_class:u3, boundary_distance:u3, agreement:u1)",
            "derived_contract": "F(M, CPR1/dxi, X_<s, s) -> (class:u3, boundary_distance:u3, agreement:u1), deterministic and video-independent code with zero new payload",
        },
        "base": {"total_bits": base_total_bits, "wrong_bits": base_wrong_bits},
        "gf1": {
            "model": gf1_model,
            "total_gain_screen_bits": gf1_total_gain,
            "wrong_gain_screen_bits": gf1_wrong_gain,
        },
        "overlap_method": {
            "definition": "cross-fitted surrogate gain divided by cross-fitted GF1 gain, evaluated on the same wrong sites and clipped to [0,1]",
            "model_family": "one categorical log-odds offset for each (generated_class,hpac_model_class,boundary_distance,agreement) cell; 250 cells for GF1 and every surrogate",
            "why_not_joint_2500_cells": "a route-by-GF1 joint table changes model complexity and can make held-out residual gain negative; it cannot establish derivability",
            "thresholds": {"closed_below": 0.10, "derived_at_or_above": 0.90},
        },
        "routes": routes,
        "typed_verdict": verdict,
        "best_route": best_route,
        "best_wrong_overlap_fraction": best_overlap,
        "verdict_reason": reason,
        "typed_implication": {
            "original_candidates_1_and_2": "not derived; their exact GF1 tuple still requires the counted packet",
            "partial_variant": "replace generated_class with the already-available HPAC argmax and causal predicted-boundary state; zero new payload but only the measured partial predictive overlap",
            "partial_variant_replacement_required_bytes": 42_017,
            "partial_variant_replacement_fraction_of_current_pool": 42_017 / 126_926,
            "original_packet_variant_replacement_required_bytes": 89_620,
            "original_packet_variant_replacement_fraction_of_current_pool": 89_620 / 126_926,
            "joint_price_required": True,
            "warning": "the M-derived state overlaps shipped mixer/corrector state, so this SCREEN cannot establish an incremental physical-byte win",
        },
        "packet_arithmetic": {
            "current_g_plus_m_pool_bytes": 126_926,
            "largest_passing_pool_bytes": 84_909,
            "gf1_packet_bytes": 47_603,
            "allowance_after_packet_bytes": 37_306,
            "replacement_required_with_packet_bytes": 89_620,
            "replacement_required_with_zero_byte_surrogate_bytes": 42_017,
            "note": "overlap fraction is predictive overlap, not packet compressibility; it cannot be multiplied by 47,603 to invent a residual packet price",
        },
        "boundaries": [
            "No real coder, archive build, scorer, Modal, Metal, or authority evaluation ran.",
            "GF1 uses an older LB1-lineage fit while X/pmax are the live cc10 field lineage named by the charter.",
            "A SCREEN closure applies to the four declared causal surrogate formulations and candidates 1+2 at the unchanged GF1 packet charge; it is not a theorem over all conceivable decoder algorithms.",
        ],
    }
    result_path = root / "RESULT.json"
    atomic_json(result_path, result)
    completion = {
        "schema": SCHEMA,
        "stage": "analyze",
        "result": file_fact(result_path),
        "typed_verdict": verdict,
        "best_route": best_route,
        "best_wrong_overlap_fraction": best_overlap,
    }
    atomic_json(root / "stage_02_analyze_complete.json", completion)
    del fields
    return result


def run_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "MANIFEST.json"
    rows = [
        file_fact(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    manifest = {
        "schema": SCHEMA,
        "stage": "manifest",
        "root": str(root),
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "exclusion": "MANIFEST.json is self-referential and omitted",
    }
    atomic_json(manifest_path, manifest)
    return manifest


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("materialize", "analyze", "manifest", "all"), required=True)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--minimum-free-bytes", type=int, default=2 * 1024**3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    free = os.statvfs(args.store.parent)
    free_bytes = free.f_bavail * free.f_frsize
    if free_bytes < args.minimum_free_bytes:
        raise Dds1Error(
            f"Vertigo storage preflight failed: free={free_bytes}, required={args.minimum_free_bytes}"
        )
    args.store.mkdir(parents=True, exist_ok=True)
    lock_path = args.store / "RUN.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Dds1Error(f"another DDS1 process holds {lock_path}") from exc

        started = time.time()
        if args.stage in ("materialize", "all"):
            materialized = run_materialize(args.store, args.seed)
            print(json.dumps({"materialize": materialized}, indent=2), flush=True)
        if args.stage in ("analyze", "all"):
            result = run_analyze(args.store, args.seed)
            print(json.dumps({"analyze": result}, indent=2), flush=True)
        if args.stage in ("manifest", "all"):
            manifest = run_manifest(args.store)
            print(json.dumps({"manifest": manifest}, indent=2), flush=True)
        print(json.dumps({"elapsed_seconds": time.time() - started}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
