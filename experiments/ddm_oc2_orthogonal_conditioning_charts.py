#!/usr/bin/env python3
"""Measure decode-derived conditioning charts orthogonal to lb1 ``patch192``.

The two candidate charts are pre-registered constants in this module.  The
``screen`` stage measures their held-out indicator-ledger overlap with
``patch192``; the ``prepare`` stage installs an admitted chart set into the
retained lb1 generation-22 runtime.  Exact archive deltas are produced by the
existing resumable ``ddm_jg2_tail_reencode`` encoder, never by this module.

Every chart is causal and receiver-derived.  It changes coder probabilities,
not decoded tokens, and stores no video-derived parameter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.ddm_fcd1_field_for_coder_diagonal import (
    NATIVE_CORRECTOR_BUILD,
    PYTHON_CORRECTOR_SELECTION,
    stage_runtime,
)
from experiments.ddm_jg2_tail_reencode import (
    RESIDUAL_COMPACT_BYTES,
    read_archive_member,
    split_member,
)
from experiments.ddm_mi1_indicator_model_axis import _code_bits, _fit_offsets
from tac.candidate_seal import CONSISTENT, check_pin_consistency

AP_ROOT = Path("/Volumes/APDataStore/pact")
VERTIGO_ROOT = Path("/Volumes/VertigoDataTier/pact")
STORE = AP_ROOT / "ddm_oc2_orthogonal_conditioning_charts"
MEASUREMENT = STORE / "measurement_v1"
RETAINED = STORE / "retained"

LB1_ROOT = AP_ROOT / "ddm_lb1_banked_lossless_joint_collect"
LB1_NATIVE_RUNTIME = LB1_ROOT / "runtime_candidate_native"
LB1_ARCHIVE = LB1_ROOT / "retained" / "candidate_lb1_joint22_patch192.zip"
LB1_RECEIPT = LB1_ROOT / "retained" / "S1_encode_lb1_joint22_patch192.json"
LB1_NATIVE_IDENTITY = LB1_ROOT / "measurement_v1" / "NATIVE_IDENTITY.json"
JT21_ROOT = AP_ROOT / "ddm_gb1_groupbin8_conditioning"
JT21_RUNTIME = JT21_ROOT / "runtime_joint21"
JT21_ARCHIVE = JT21_ROOT / "retained" / "candidate_gb1_joint21.zip"
JT21_CONTROL = LB1_ROOT / "retained" / "S1_control_600.json"
RANK_SOLO_STORE = STORE / "rank_solo"
RANK_AFTER_STORE = STORE / "rank_after_patch"

MI1_VERIFY = (
    AP_ROOT / "ddm_mi1_indicator_model_axis" / "measurement_v1" / "VERIFY.json"
)
DF1_FIELDS = AP_ROOT / "ddm_df1_dddb_field" / "measurement_v1" / "retained" / "fields"
DF1_ARGMAX = DF1_FIELDS / "position_coding_argmax.u8.bin"
DF1_PMAX = DF1_FIELDS / "position_coding_pmax.f32le.bin"
TOKENS = (
    VERTIGO_ROOT
    / "ddm_to2_token_ordering_race"
    / "measurement_v1"
    / "retained"
    / "input"
    / "dx2_tokens_decoded.u8"
)

N = 600
HEIGHT = 384
WIDTH = 512
PLANE = HEIGHT * WIDTH
POSITIONS = N * PLANE
NUM_CLASSES = 5
UNKNOWN = NUM_CLASSES
LN2 = math.log(2.0)
MINIMUM_FREE_BYTES = 8 << 30
S_PER_BYTE = 25.0 / 37_545_489.0
ORTHOGONALITY_CUTOFF = 0.44
SOLO_FIRE_BAR_BYTES = 30

PINS: dict[str, tuple[Path, int, str]] = {
    "lb1_archive": (
        LB1_ARCHIVE,
        180_083,
        "5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9",
    ),
    "lb1_receipt": (
        LB1_RECEIPT,
        2_052,
        "575f423c4c952933f48a3245b7d4e20ffa8a0f4fa02419ac544f8164fb26556c",
    ),
    "tokens": (
        TOKENS,
        117_964_800,
        "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb",
    ),
    "coding_argmax": (
        DF1_ARGMAX,
        117_964_800,
        "db498280c22c3aa1b787310e25435116911933216cae558f309f8b10baf7994e",
    ),
    "coding_pmax": (
        DF1_PMAX,
        471_859_200,
        "f37e3d8a21d02647437bf950d7a8a75b751c2a9644c7b8ad48aca2833be4794b",
    ),
    "jt21_archive": (
        JT21_ARCHIVE,
        180_192,
        "ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3",
    ),
}

# These predictions are deliberately written before ``screen`` touches any
# candidate outcome.  They are structural priors, not fitted numbers.
CHARTS: dict[str, dict[str, Any]] = {
    "temporal_transition30": {
        "axis": "temporal class transition: current predicted class x previous decoded-frame class",
        "cells": NUM_CLASSES * (NUM_CLASSES + 1),
        "predicted_patch192_overlap_fraction": 0.20,
        "prediction_reason": (
            "the chart has no x/y coordinate and refines exact temporal class identity "
            "beyond the shipped agree bit; scene stationarity can still induce some overlap"
        ),
    },
    "causal_edge30": {
        "axis": "local class-conditioned structure: current predicted class x first available causal-neighbour class",
        "cells": NUM_CLASSES * (NUM_CLASSES + 1),
        "predicted_patch192_overlap_fraction": 0.35,
        "prediction_reason": (
            "the chart has no absolute coordinate and exposes class-pair asymmetry hidden "
            "by count-only spatial4/homogeneity; road/lane geography can correlate it with patches"
        ),
    },
}

CHART_SETS: dict[str, tuple[str, ...]] = {
    "temporal": ("temporal_transition30",),
    "edge": ("causal_edge30",),
    "joint": ("temporal_transition30", "causal_edge30"),
    "rank": ("miss_rank8",),
}

RANK_CHART: dict[str, Any] = {
    "axis": "coefficient magnitude-rank: runner-up probability as a share of total non-argmax mass",
    "cells": 8,
    "predicted_patch192_overlap_fraction": 0.15,
    "prediction_reason": (
        "the chart uses only the five-symbol probability row and no coordinate, decoded class, "
        "or frame position; correlation can remain because confidence geometry is scene-dependent"
    ),
}


class Oc2Error(RuntimeError):
    """A custody, chart, runtime, or identity refusal."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def require_pin(key: str) -> dict[str, Any]:
    path, expected_bytes, expected_sha = PINS[key]
    if not path.is_file():
        raise Oc2Error(f"missing pinned input {key}: {path}")
    observed = file_fact(path)
    if observed["bytes"] != expected_bytes or observed["sha256"] != expected_sha:
        raise Oc2Error(
            f"pin mismatch for {key}: {observed['bytes']}/{observed['sha256']} "
            f"!= {expected_bytes}/{expected_sha}"
        )
    return observed


def rederive_headroom() -> dict[str, Any]:
    verify = json.loads(MI1_VERIFY.read_text())
    indicator = float(verify["indicator"]["bytes"])
    entropy = float(verify["model_entropy"]["bytes"])
    excess = indicator - entropy
    declared = float(verify["realised_excess_over_entropy"]["bytes"])
    positions = int(verify["positions"])
    if positions != POSITIONS or not math.isclose(excess, declared, abs_tol=1e-9):
        raise Oc2Error("mi1 headroom does not re-derive from its source components")

    lb1 = json.loads(LB1_RECEIPT.read_text())
    if lb1.get("tokens_changed") != 0 or lb1.get("archive_bytes_candidate") != 180_083:
        raise Oc2Error("lb1 receipt no longer describes the retained lossless candidate")

    # Current-body accounting is a single physical lineage, never a sum of
    # standalone ledger credits: dx2 180368 -> gb1 180215 -> jt21 180192 -> lb1 180083.
    chain = [
        {"collection": "gb1", "before_bytes": 180_368, "after_bytes": 180_215, "marginal_bytes": 153},
        {"collection": "jt21", "before_bytes": 180_215, "after_bytes": 180_192, "marginal_bytes": 23},
        {"collection": "lb1", "before_bytes": 180_192, "after_bytes": 180_083, "marginal_bytes": 109},
    ]
    cumulative = 180_368 - 180_083
    if cumulative != sum(int(row["marginal_bytes"]) for row in chain):
        raise Oc2Error("current-body physical collection chain does not close")
    remaining = excess - cumulative
    return {
        "denominator": {
            "positions": positions,
            "definition": "all n600 384x512 token positions on the dx2/lb1 decoded field",
        },
        "mi1": {
            "indicator_bytes": indicator,
            "model_entropy_bytes": entropy,
            "realised_excess_bytes": excess,
            "z": float(verify["realised_excess_over_entropy"]["z"]),
            "source": file_fact(MI1_VERIFY),
        },
        "current_physical_chain": chain,
        "current_body_cumulative_collected_bytes": cumulative,
        "current_body_remaining_proxy_bytes": remaining,
        "current_body_remaining_fraction_of_mi1_excess": remaining / excess,
        "jt22_off_branch": {
            "measured_bytes": 1,
            "disposition": (
                "separate dx2 mixer-context race; evidence of overlap, not present in lb1 and "
                "therefore not added to the current-body subtraction"
            ),
        },
        "boundary": (
            "mi1 excess is an indicator-ledger target while the chain is exact archive bytes; "
            "the subtraction is the campaign's conditioning-mass proxy, not an information-theoretic hard bound"
        ),
    }


def stage_preflight() -> dict[str, Any]:
    STORE.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(STORE).free
    if free < MINIMUM_FREE_BYTES:
        raise Oc2Error(f"APDataStore free {free} B < required {MINIMUM_FREE_BYTES} B")
    inputs = {key: require_pin(key) for key in PINS}
    source_verdict = check_pin_consistency(LB1_NATIVE_RUNTIME)
    if source_verdict.verdict != CONSISTENT:
        raise Oc2Error(f"lb1 generation-22 runtime is pin-inconsistent: {source_verdict.summary()}")
    identity = json.loads(LB1_NATIVE_IDENTITY.read_text())
    if not identity.get("result", {}).get("byte_identical"):
        raise Oc2Error("lb1 generation-22 native identity receipt is not a passing source")
    payload = {
        "schema": "ddm_oc2_preflight.v1",
        "axis": "[macOS-CPU advisory / scorer-free source re-derivation]",
        "score_claim": False,
        "promotable": False,
        "storage": {
            "path": str(STORE),
            "free_bytes": free,
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
            "status": "PASS",
        },
        "inputs": inputs,
        "source_runtime": {
            "path": str(LB1_NATIVE_RUNTIME),
            "pin_consistency": source_verdict.verdict,
            "generation": 22,
            "native_identity": file_fact(LB1_NATIVE_IDENTITY),
        },
        "headroom": rederive_headroom(),
        "preregistered_charts": CHARTS,
        "orthogonality_cutoff_fraction": ORTHOGONALITY_CUTOFF,
        "solo_fire_bar_bytes": SOLO_FIRE_BAR_BYTES,
        "prediction_frozen_before_screen": True,
    }
    atomic_json(MEASUREMENT / "PREFLIGHT_AND_PREREGISTRATION.json", payload)
    return payload


def _causal_anchor_sources() -> np.ndarray:
    """First available neighbour under the exact 190-group decoder schedule."""
    y, x = np.indices((HEIGHT, WIDTH), dtype=np.int64)
    flat = (y * WIDTH + x).reshape(-1)
    group = ((x % 64) + 2 * (y % 64)).reshape(-1)
    anchor = np.full(PLANE, -1, dtype=np.int64)
    for dx, dy in ((-1, 0), (0, -1), (1, -1), (-1, -1)):
        nx = x.reshape(-1) + dx
        ny = y.reshape(-1) + dy
        inside = (nx >= 0) & (nx < WIDTH) & (ny >= 0) & (ny < HEIGHT)
        source = np.clip(ny, 0, HEIGHT - 1) * WIDTH + np.clip(nx, 0, WIDTH - 1)
        available = inside & (group[source] < group[flat])
        take = (anchor < 0) & available
        anchor[take] = source[take]
    return anchor


def build_screen_compact() -> dict[str, np.ndarray]:
    tokens = np.memmap(TOKENS, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    argmax = np.memmap(DF1_ARGMAX, dtype=np.uint8, mode="r", shape=(N, PLANE))
    pmax = np.memmap(DF1_PMAX, dtype="<f4", mode="r", shape=(N, PLANE))
    y, x = np.indices((HEIGHT, WIDTH), dtype=np.int64)
    patch192 = ((y // 32) * (WIDTH // 32) + x // 32).reshape(-1).astype(np.uint8)
    anchor_source = _causal_anchor_sources()
    anchor_present = anchor_source >= 0
    safe_anchor = np.maximum(anchor_source, 0)

    chunks: dict[str, list[np.ndarray]] = {
        "logit": [],
        "flip": [],
        "patch192": [],
        "temporal_transition30": [],
        "causal_edge30": [],
    }
    prev1 = np.full(PLANE, UNKNOWN, dtype=np.uint8)
    for frame in range(N):
        token = np.asarray(tokens[frame], dtype=np.uint8).reshape(-1)
        arg = np.asarray(argmax[frame], dtype=np.uint8)
        q = 1.0 - np.asarray(pmax[frame], dtype=np.float64)
        live = q > 0.0
        anchor_class = np.full(PLANE, UNKNOWN, dtype=np.uint8)
        anchor_class[anchor_present] = token[safe_anchor[anchor_present]]

        ql = q[live]
        chunks["logit"].append(np.log(ql) - np.log1p(-ql))
        chunks["flip"].append((token != arg)[live].astype(np.uint8))
        chunks["patch192"].append(patch192[live])
        chunks["temporal_transition30"].append(
            (arg.astype(np.int64) * (NUM_CLASSES + 1) + prev1.astype(np.int64))[live].astype(np.uint8)
        )
        chunks["causal_edge30"].append(
            (arg.astype(np.int64) * (NUM_CLASSES + 1) + anchor_class.astype(np.int64))[live].astype(np.uint8)
        )
        prev1 = token.copy()
    return {key: np.concatenate(parts) for key, parts in chunks.items()}


def _heldout_gain(
    logit: np.ndarray,
    flip: np.ndarray,
    cell: np.ndarray,
    n_cells: int,
    fold_a: np.ndarray,
) -> dict[str, float]:
    fold_b = ~fold_a
    beta_a = _fit_offsets(logit, flip, cell, n_cells, fold_a)
    beta_b = _fit_offsets(logit, flip, cell, n_cells, fold_b)
    heldout = np.empty(logit.size, dtype=np.float64)
    heldout[fold_b] = beta_a[cell[fold_b]]
    heldout[fold_a] = beta_b[cell[fold_a]]
    base_bits = _code_bits(logit, flip, np.zeros(logit.size, dtype=np.float64))
    measured_bits = _code_bits(logit, flip, heldout)
    return {
        "gain_bits": base_bits - measured_bits,
        "gain_bytes": (base_bits - measured_bits) / 8.0,
        "max_abs_fold_offset": max(float(np.max(np.abs(beta_a))), float(np.max(np.abs(beta_b)))),
    }


def stage_screen(seed: int) -> dict[str, Any]:
    prereg = MEASUREMENT / "PREFLIGHT_AND_PREREGISTRATION.json"
    if not prereg.is_file():
        raise Oc2Error("run --stage preflight before screen; predictions must exist first")
    started = time.time()
    compact = build_screen_compact()
    logit = compact["logit"]
    flip = compact["flip"]
    rng = np.random.default_rng(seed)
    fold_a = rng.integers(0, 2, size=logit.size, endpoint=False).astype(bool)
    patch = compact["patch192"].astype(np.int64)
    patch_result = _heldout_gain(logit, flip, patch, 192, fold_a)

    rows = []
    for name, spec in CHARTS.items():
        chart = compact[name].astype(np.int64)
        n_cells = int(spec["cells"])
        solo = _heldout_gain(logit, flip, chart, n_cells, fold_a)
        joint_cell = patch * n_cells + chart
        joint = _heldout_gain(logit, flip, joint_cell, 192 * n_cells, fold_a)
        marginal_after_patch = joint["gain_bytes"] - patch_result["gain_bytes"]
        measured_overlap = (
            1.0 - marginal_after_patch / solo["gain_bytes"]
            if solo["gain_bytes"] > 0.0
            else None
        )
        rows.append(
            {
                "chart": name,
                "axis": spec["axis"],
                "cells": n_cells,
                "predicted_patch192_overlap_fraction": spec[
                    "predicted_patch192_overlap_fraction"
                ],
                "solo_heldout_gain_bytes": solo["gain_bytes"],
                "patch192_heldout_gain_bytes": patch_result["gain_bytes"],
                "joint_heldout_gain_bytes": joint["gain_bytes"],
                "marginal_heldout_gain_after_patch_bytes": marginal_after_patch,
                "measured_patch192_overlap_fraction": measured_overlap,
                "orthogonal_under_cutoff": bool(
                    measured_overlap is not None
                    and math.isfinite(measured_overlap)
                    and measured_overlap <= ORTHOGONALITY_CUTOFF
                ),
                "screen_axis": "[macOS-CPU advisory / random 2-fold held-out indicator ledger]",
                "physical_bytes": "UNMEASURED_UNTIL_REAL_JOINT_REENCODE",
            }
        )
    payload = {
        "schema": "ddm_oc2_chart_screen.v1",
        "axis": "[macOS-CPU advisory / scorer-free held-out indicator ledger]",
        "score_claim": False,
        "promotable": False,
        "seed": seed,
        "population": {"n_positions_live": int(logit.size), "n_positions_total": POSITIONS},
        "preregistration": file_fact(prereg),
        "patch192": patch_result,
        "orthogonality_cutoff_fraction": ORTHOGONALITY_CUTOFF,
        "rows": rows,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(MEASUREMENT / "CHART_SCREEN.json", payload)
    return payload


def stage_preregister_rank() -> dict[str, Any]:
    screen_path = MEASUREMENT / "CHART_SCREEN.json"
    if not screen_path.is_file():
        raise Oc2Error("rank-axis continuation requires the original chart screen receipt")
    screen = json.loads(screen_path.read_text())
    original_closed = all(not bool(row["orthogonal_under_cutoff"]) for row in screen["rows"])
    if not original_closed:
        raise Oc2Error("rank continuation is only licensed after the original chart set closes")
    payload = {
        "schema": "ddm_oc2_rank_preregistration.v1",
        "axis": "[preregistered before miss_rank8 measurement]",
        "score_claim": False,
        "promotable": False,
        "chart": "miss_rank8",
        **RANK_CHART,
        "orthogonality_cutoff_fraction": ORTHOGONALITY_CUTOFF,
        "solo_fire_bar_bytes_after_patch192": SOLO_FIRE_BAR_BYTES,
        "measurement": (
            "one real n600 re-encode on jt21 without patch192 and one real n600 re-encode "
            "on lb1 with patch192; exact overlap = 1 - after_patch_saving/solo_saving"
        ),
        "continuation_reason": (
            "the original temporal and local-class charts closed; magnitude-rank is a distinct "
            "charter-named structural axis, not a retuning of either failed chart"
        ),
        "original_screen": file_fact(screen_path),
        "prediction_frozen_before_rank_encode": True,
    }
    atomic_json(MEASUREMENT / "RANK_PREREGISTRATION.json", payload)
    return payload


def replace_once(text: str, before: str, after: str, *, label: str) -> str:
    count = text.count(before)
    if count != 1:
        raise Oc2Error(f"runtime patch anchor {label!r} occurs {count} times")
    return text.replace(before, after, 1)


def replace_first(text: str, before: str, after: str, *, label: str) -> str:
    count = text.count(before)
    if count < 1:
        raise Oc2Error(f"runtime patch anchor {label!r} is absent")
    return text.replace(before, after, 1)


PY_FEATURE_PATCH = (
    '            "prev1_cls": (\n'
    "                self.prev1[flat].astype(np.int64)\n"
    "                if self.have_prev\n"
    "                else np.full(flat.size, NUM_CLASSES, dtype=np.int64)\n"
    "            ),\n"
    '            "causal_anchor": (\n'
    "                np.where(\n"
    "                    available[0], classes[0],\n"
    "                    np.where(\n"
    "                        available[1], classes[1],\n"
    "                        np.where(\n"
    "                            available[2], classes[2],\n"
    "                            np.where(available[3], classes[3], NUM_CLASSES),\n"
    "                        ),\n"
    "                    ),\n"
    "                ).astype(np.int64)\n"
    "            ),\n"
)


def patch_python_runtime(runtime: Path, charts: tuple[str, ...]) -> dict[str, Any]:
    target = runtime / "runtime" / "fx2_model_axis_corrector.py"
    before = file_fact(target)
    text = target.read_text()
    if '"prev1_cls"' not in text:
        text = replace_once(
            text,
            '            "patch192": (\n',
            PY_FEATURE_PATCH + '            "patch192": (\n',
            label="candidate features",
        )
    if "temporal_transition30" in charts and 'def temporal_transition30(f):' not in text:
        addition = (
            "    def temporal_transition30(f):\n"
            '        return f["cls"] * (NUM_CLASSES + 1) + f["prev1_cls"]\n\n'
            '    specs.update({"temporal_transition30": (NUM_CLASSES * (NUM_CLASSES + 1), temporal_transition30)})\n\n'
        )
        text = replace_once(text, "    def groupbin8_only(f):\n", addition + "    def groupbin8_only(f):\n", label="temporal family")
    if "causal_edge30" in charts and 'def causal_edge30(f):' not in text:
        addition = (
            "    def causal_edge30(f):\n"
            '        return f["cls"] * (NUM_CLASSES + 1) + f["causal_anchor"]\n\n'
            '    specs.update({"causal_edge30": (NUM_CLASSES * (NUM_CLASSES + 1), causal_edge30)})\n\n'
        )
        text = replace_once(text, "    def groupbin8_only(f):\n", addition + "    def groupbin8_only(f):\n", label="edge family")
    missing_members = [chart for chart in charts if f'        "{chart}",\n' not in text]
    if missing_members:
        if tuple(missing_members) != charts:
            raise Oc2Error("runtime contains only a partial oc2 chart set")
        text = replace_once(
            text,
            '        "patch192_only",\n',
            '        "patch192_only",\n'
            + "".join(f'        "{chart}",\n' for chart in charts),
            label="ordered config members",
        )
    temporary = target.with_suffix(".py.partial")
    temporary.write_text(text)
    os.replace(temporary, target)

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,sys; "
                f"sys.path.insert(0,{str(runtime)!r}); "
                "from runtime.fx2_model_axis_corrector import fx2_family_specs; "
                "from runtime.free_corrector import SHIPPED_CONFIG; "
                "s=fx2_family_specs(); print(json.dumps({'families':list(SHIPPED_CONFIG['families']), "
                "'sizes':{k:int(s[k][0]) for k in SHIPPED_CONFIG['families']}}))"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise Oc2Error(f"patched Python runtime import failed:\n{probe.stderr}")
    wiring = json.loads(probe.stdout.strip().splitlines()[-1])
    expected_count = 22 + len(charts)
    if len(wiring["families"]) != expected_count:
        raise Oc2Error(f"family count {len(wiring['families'])} != {expected_count}")
    for chart in charts:
        if wiring["sizes"].get(chart) != int(CHARTS[chart]["cells"]):
            raise Oc2Error(f"chart wiring mismatch for {chart}: {wiring['sizes'].get(chart)}")
    return {"before": before, "after": file_fact(target), "wiring": wiring}


def patch_rank_python_runtime(runtime: Path) -> dict[str, Any]:
    """Add the exact pre-mixer coefficient-rank chart to a 21- or 22-family body."""
    target = runtime / "runtime" / "fx2_model_axis_corrector.py"
    before = file_fact(target)
    text = target.read_text()
    if 'def miss_rank8(f):' not in text:
        rank_calc = (
            "        rank_rows = np.asarray(probability, dtype=np.float64)\n"
            "        rank_residual = rank_rows.copy()\n"
            "        rank_residual[np.arange(rank_rows.shape[0]), state.arg] = -np.inf\n"
            "        rank_second = rank_residual.max(axis=1)\n"
            "        miss_rank8 = np.clip(\n"
            "            (rank_second * 8.0 / state.one_minus).astype(np.int64), 0, 7\n"
            "        )\n\n"
        )
        text = replace_once(
            text,
            "        features = {\n",
            rank_calc + "        features = {\n",
            label="rank feature calculation",
        )
        text = replace_once(
            text,
            '            "groupbin8": (\n',
            '            "miss_rank8": miss_rank8,\n            "groupbin8": (\n',
            label="rank feature dict",
        )
        rank_spec = (
            "    def miss_rank8(f):\n"
            '        return f["miss_rank8"]\n\n'
            '    specs.update({"miss_rank8": (8, miss_rank8)})\n\n'
        )
        text = replace_once(
            text,
            "    def groupbin8_only(f):\n",
            rank_spec + "    def groupbin8_only(f):\n",
            label="rank family spec",
        )
        if '        "patch192_only",\n' in text:
            member_anchor = '        "patch192_only",\n'
        else:
            member_anchor = '        "cls_groupbin8",\n'
        text = replace_once(
            text,
            member_anchor,
            member_anchor + '        "miss_rank8",\n',
            label="rank config member",
        )
        temporary = target.with_suffix(".py.partial")
        temporary.write_text(text)
        os.replace(temporary, target)

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,sys; "
                f"sys.path.insert(0,{str(runtime)!r}); "
                "from runtime.fx2_model_axis_corrector import fx2_family_specs; "
                "from runtime.free_corrector import SHIPPED_CONFIG; "
                "s=fx2_family_specs(); print(json.dumps({'families':list(SHIPPED_CONFIG['families']), "
                "'size':int(s['miss_rank8'][0])}))"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise Oc2Error(f"rank runtime import failed:\n{probe.stderr}")
    wiring = json.loads(probe.stdout.strip().splitlines()[-1])
    if wiring["size"] != 8 or wiring["families"][-1] != "miss_rank8":
        raise Oc2Error(f"rank runtime wiring mismatch: {wiring}")
    return {"before": before, "after": file_fact(target), "wiring": wiring}


def _rank_base(base: str) -> tuple[Path, Path, Path, str]:
    if base == "solo":
        return JT21_RUNTIME, JT21_ARCHIVE, RANK_SOLO_STORE, "oc2_rank_solo"
    if base == "after":
        return LB1_NATIVE_RUNTIME, LB1_ARCHIVE, RANK_AFTER_STORE, "oc2_rank_after"
    raise Oc2Error(f"unknown rank base {base!r}")


def stage_prepare_rank(base: str) -> dict[str, Any]:
    prereg = MEASUREMENT / "RANK_PREREGISTRATION.json"
    if not prereg.is_file():
        raise Oc2Error("rank chart must be preregistered before runtime preparation")
    source_runtime, pointer_archive, rank_store, tag = _rank_base(base)
    runtime = rank_store / "runtime_candidate"
    rank_store.mkdir(parents=True, exist_ok=True)
    staged = stage_runtime(source_runtime, pointer_archive, runtime)
    patched = patch_rank_python_runtime(runtime)
    verdict = check_pin_consistency(runtime)
    if verdict.verdict != CONSISTENT:
        raise Oc2Error(f"rank runtime is pin-inconsistent: {verdict.summary()}")
    command = [
        ".venv/bin/python",
        "experiments/ddm_jg2_tail_reencode.py",
        "--stage",
        "encode",
        "--store",
        str(rank_store),
        "--runtime-root",
        str(runtime),
        "--pointer-archive",
        str(pointer_archive),
        "--expect-pointer-sha256",
        sha256_file(pointer_archive),
        "--tokens",
        str(TOKENS),
        "--tag",
        tag,
        "--frames",
        "600",
        "--checkpoint-every",
        "25",
        "--resume",
    ]
    payload = {
        "schema": "ddm_oc2_rank_runtime_prepare.v1",
        "axis": "[build artifact / no score claim]",
        "score_claim": False,
        "promotable": False,
        "base": base,
        "preregistration": file_fact(prereg),
        "runtime": staged,
        "patch": patched,
        "pin_consistency": verdict.verdict,
        "encode_command": command,
        "retention": (
            "the resumable encoder retains corrector state, per-frame ledger, stream, "
            "candidate archive, receipt, and build products under this base-specific store"
        ),
    }
    atomic_json(rank_store / "RUNTIME_PREPARE.json", payload)
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _seed_one_rank_control(base: str) -> dict[str, Any]:
    _source_runtime, pointer_archive, rank_store, _tag = _rank_base(base)
    sections = split_member(read_archive_member(pointer_archive))
    token_stream = sections["tail"][RESIDUAL_COMPACT_BYTES:]
    if base == "solo":
        source_path = JT21_CONTROL
        source = json.loads(source_path.read_text())
        expected_sha = str(source["emitted_sha256"])
        expected_bytes = int(source["emitted_bytes"])
    else:
        source_path = LB1_RECEIPT
        source = json.loads(source_path.read_text())
        expected_sha = str(source["stream"]["sha256"])
        expected_bytes = int(source["stream"]["bytes"])
    observed_sha = _sha256_bytes(token_stream)
    identical = len(token_stream) == expected_bytes and observed_sha == expected_sha
    if not identical:
        raise Oc2Error(f"retained n600 control does not match {base} pointer token stream")
    payload = {
        "schema": "ddm_oc2_reused_n600_control.v1",
        "axis": "[macOS-CPU advisory / scorer-free exact retained control reuse]",
        "score_claim": False,
        "promotable": False,
        "base": base,
        "byte_identical": True,
        "frames": N,
        "full_run": True,
        "emitted_bytes": len(token_stream),
        "emitted_sha256": observed_sha,
        "shipped_token_stream_sha256": observed_sha,
        "pointer_archive": file_fact(pointer_archive),
        "source_control_receipt": file_fact(source_path),
        "source_full_n600_reencode": True,
        "fresh_control_run": False,
        "reuse_scope": (
            "same exact pointer body, token field, RC64 encoder, and already-proven n600 "
            "inverse; miss_rank8 changes the probability law, so any admitted candidate "
            "still requires a full receiver decode-identity receipt"
        ),
        "payload_custody": (
            "the exact stream remains inside the pointer archive and at the source receipt's "
            "retained stream path; only the proven receipt is consumed here"
        ),
    }
    destination = rank_store / "retained" / "S1_control_600.json"
    atomic_json(destination, payload)
    return {"receipt": file_fact(destination), "payload": payload}


def stage_seed_rank_controls() -> dict[str, Any]:
    payload = {
        "schema": "ddm_oc2_rank_controls.v1",
        "axis": "[scorer-free exact retained control reuse]",
        "score_claim": False,
        "promotable": False,
        "solo": _seed_one_rank_control("solo"),
        "after": _seed_one_rank_control("after"),
    }
    atomic_json(MEASUREMENT / "RANK_CONTROLS.json", payload)
    return payload


def stage_adjudicate_rank() -> dict[str, Any]:
    receipts = {}
    for base, rank_store, tag in (
        ("solo", RANK_SOLO_STORE, "oc2_rank_solo"),
        ("after", RANK_AFTER_STORE, "oc2_rank_after"),
    ):
        path = rank_store / "retained" / f"S1_encode_{tag}.json"
        if not path.is_file():
            raise Oc2Error(f"rank encode receipt missing: {path}")
        row = json.loads(path.read_text())
        if (
            row.get("tokens_changed") != 0
            or int(row.get("frames", 0)) != N
            or not row.get("delta_trustworthy")
        ):
            raise Oc2Error(f"rank {base} receipt is not a trustworthy lossless row")
        candidate = Path(row["candidate_archive"]["path"])
        if file_fact(candidate) != row["candidate_archive"]:
            raise Oc2Error(f"rank {base} retained candidate archive drifted")
        expected_pointer = JT21_ARCHIVE if base == "solo" else LB1_ARCHIVE
        if row.get("pointer_archive") != file_fact(expected_pointer):
            raise Oc2Error(f"rank {base} receipt is not based on its pinned pointer")
        receipts[base] = {"path": path, "fact": file_fact(path), "row": row}
    solo_saving = -int(receipts["solo"]["row"]["archive_delta_bytes"])
    after_saving = -int(receipts["after"]["row"]["archive_delta_bytes"])
    overlap = 1.0 - after_saving / solo_saving if solo_saving > 0 else None
    orthogonal = bool(
        overlap is not None
        and math.isfinite(overlap)
        and overlap <= ORTHOGONALITY_CUTOFF
    )
    if after_saving >= SOLO_FIRE_BAR_BYTES:
        disposition = "ADMITTED__SEAL_FIRE_ORDER_OWED"
    elif after_saving > 0:
        disposition = "BANKED_BELOW_SOLO_FIRE_BAR"
    else:
        disposition = "CLOSED_NONPOSITIVE_MARGINAL"
    payload = {
        "schema": "ddm_oc2_rank_adjudication.v1",
        "axis": "[macOS-CPU advisory / scorer-free exact byte measurement]",
        "score_claim": False,
        "promotable": False,
        "chart": "miss_rank8",
        "predicted_patch192_overlap_fraction": RANK_CHART[
            "predicted_patch192_overlap_fraction"
        ],
        "solo_jt21": {
            "saving_bytes": solo_saving,
            "candidate": receipts["solo"]["row"]["candidate_archive"],
            "receipt": receipts["solo"]["fact"],
        },
        "after_patch192_lb1": {
            "marginal_saving_bytes": after_saving,
            "delta_S_rate": -after_saving * S_PER_BYTE,
            "candidate": receipts["after"]["row"]["candidate_archive"],
            "receipt": receipts["after"]["fact"],
        },
        "measured_patch192_overlap_fraction": overlap,
        "orthogonality_cutoff_fraction": ORTHOGONALITY_CUTOFF,
        "orthogonal_under_cutoff": orthogonal,
        "solo_fire_bar_bytes": SOLO_FIRE_BAR_BYTES,
        "disposition": disposition,
    }
    atomic_json(MEASUREMENT / "RANK_ADJUDICATION.json", payload)
    return payload


def runtime_for_set(set_name: str) -> Path:
    return STORE / f"runtime_{set_name}"


def tag_for_set(set_name: str) -> str:
    return f"oc2_{set_name}"


def stage_prepare(set_name: str) -> dict[str, Any]:
    if set_name not in CHART_SETS:
        raise Oc2Error(f"unknown chart set {set_name!r}")
    if set_name == "rank":
        raise Oc2Error("use --stage prepare-rank --base solo|after for miss_rank8")
    charts = CHART_SETS[set_name]
    screen_path = MEASUREMENT / "CHART_SCREEN.json"
    if not screen_path.is_file():
        raise Oc2Error("run the frozen chart screen before preparing a runtime")
    screen_rows = {
        str(row["chart"]): row for row in json.loads(screen_path.read_text())["rows"]
    }
    refused = [
        chart
        for chart in charts
        if not bool(screen_rows.get(chart, {}).get("orthogonal_under_cutoff"))
    ]
    if refused:
        raise Oc2Error(f"chart set failed the frozen orthogonality screen: {refused}")
    runtime = runtime_for_set(set_name)
    staged = stage_runtime(LB1_NATIVE_RUNTIME, LB1_ARCHIVE, runtime)
    patched = patch_python_runtime(runtime, charts)
    verdict = check_pin_consistency(runtime)
    if verdict.verdict != CONSISTENT:
        raise Oc2Error(f"prepared runtime is pin-inconsistent: {verdict.summary()}")
    command = [
        ".venv/bin/python",
        "experiments/ddm_jg2_tail_reencode.py",
        "--stage",
        "encode",
        "--store",
        str(STORE),
        "--runtime-root",
        str(runtime),
        "--pointer-archive",
        str(LB1_ARCHIVE),
        "--expect-pointer-sha256",
        PINS["lb1_archive"][2],
        "--tokens",
        str(TOKENS),
        "--tag",
        tag_for_set(set_name),
        "--frames",
        "600",
        "--checkpoint-every",
        "25",
        "--resume",
    ]
    payload = {
        "schema": "ddm_oc2_runtime_prepare.v1",
        "axis": "[build artifact / no score claim]",
        "score_claim": False,
        "promotable": False,
        "set": set_name,
        "charts": list(charts),
        "runtime": staged,
        "patch": patched,
        "pin_consistency": verdict.verdict,
        "encode_command": command,
        "retention": (
            "ddm_jg2 retains checkpointed corrector state, stream, per-frame ledger, "
            "receipt, RC64 build, and candidate archive under the oc2 store"
        ),
    }
    atomic_json(MEASUREMENT / f"RUNTIME_PREPARE_{set_name}.json", payload)
    return payload


def stage_prepare_control() -> dict[str, Any]:
    runtime = STORE / "runtime_control_lb1"
    staged = stage_runtime(LB1_NATIVE_RUNTIME, LB1_ARCHIVE, runtime)
    verdict = check_pin_consistency(runtime)
    if verdict.verdict != CONSISTENT:
        raise Oc2Error(f"control runtime is pin-inconsistent: {verdict.summary()}")
    command = [
        ".venv/bin/python",
        "experiments/ddm_jg2_tail_reencode.py",
        "--stage",
        "control",
        "--store",
        str(STORE),
        "--runtime-root",
        str(runtime),
        "--pointer-archive",
        str(LB1_ARCHIVE),
        "--expect-pointer-sha256",
        PINS["lb1_archive"][2],
        "--tokens",
        str(TOKENS),
        "--frames",
        "600",
        "--checkpoint-every",
        "25",
        "--resume",
    ]
    payload = {
        "schema": "ddm_oc2_control_prepare.v1",
        "axis": "[build artifact / no score claim]",
        "score_claim": False,
        "promotable": False,
        "runtime": staged,
        "pin_consistency": verdict.verdict,
        "control_command": command,
    }
    atomic_json(MEASUREMENT / "CONTROL_PREPARE.json", payload)
    return payload


def _append_c_array_tail(text: str, anchor: str, values: list[str], label: str) -> str:
    replacement = anchor[:-2] + ", " + ", ".join(values) + "\n};"
    return replace_once(text, anchor, replacement, label=label)


def patch_native_runtime(runtime: Path, charts: tuple[str, ...]) -> dict[str, Any]:
    """Extend the retained generation-22 C port; generation-20 is never reused."""
    c_path = runtime / "runtime" / "f26_corrector_native.c"
    gate_path = runtime / "runtime" / "native_free_corrector.py"
    source_c = LB1_NATIVE_RUNTIME / "runtime" / "f26_corrector_native.c"
    source_gate = LB1_NATIVE_RUNTIME / "runtime" / "native_free_corrector.py"
    shutil.copy2(source_c, c_path)
    shutil.copy2(source_gate, gate_path)
    c_before = file_fact(c_path)
    text = c_path.read_text()
    text = replace_once(
        text,
        "#define N_FAMILIES 22 /* ddm_lb1: +cls_groupbin8 +patch192_only */",
        f"#define N_FAMILIES {22 + len(charts)} /* ddm_oc2: generation-22 plus orthogonal charts */",
        label="native family count",
    )
    enum_rows = []
    rule_rows = []
    size_rows = []
    cases = []
    extra_args = []
    call_args = []
    feature_rows = []
    for chart in charts:
        if chart == "temporal_transition30":
            enum_rows.append("    RULE_TEMPORAL_TRANSITION30")
            rule_rows.append("    RULE_TEMPORAL_TRANSITION30       /* temporal_transition30, ddm_oc2 */")
            size_rows.append("    NUM_CLASSES * (NUM_CLASSES + 1) /* temporal_transition30 */")
            cases.append(
                "    case RULE_TEMPORAL_TRANSITION30:\n"
                "        return cls * (NUM_CLASSES + 1) + prev1_cls;\n"
            )
            extra_args.append("int64_t prev1_cls")
            call_args.append("prev1_cls")
            feature_rows.append(
                "        int64_t prev1_cls = self->have_prev ? (int64_t)self->prev1[flat] : UNKNOWN;\n"
            )
        elif chart == "causal_edge30":
            enum_rows.append("    RULE_CAUSAL_EDGE30")
            rule_rows.append("    RULE_CAUSAL_EDGE30               /* causal_edge30, ddm_oc2 */")
            size_rows.append("    NUM_CLASSES * (NUM_CLASSES + 1) /* causal_edge30 */")
            cases.append(
                "    case RULE_CAUSAL_EDGE30:\n"
                "        return cls * (NUM_CLASSES + 1) + causal_anchor;\n"
            )
            extra_args.append("int64_t causal_anchor")
            call_args.append("causal_anchor")
            feature_rows.append(
                "        int64_t causal_anchor = UNKNOWN;\n"
                "        for (int slot = N_CAUSAL - 1; slot >= 0; --slot) {\n"
                "            if (available[slot]) { causal_anchor = classes[slot]; }\n"
                "        }\n"
            )
        elif chart == "miss_rank8":
            enum_rows.append("    RULE_MISS_RANK8")
            rule_rows.append("    RULE_MISS_RANK8                  /* miss_rank8, ddm_oc2 */")
            size_rows.append("    8                              /* miss_rank8 */")
            cases.append(
                "    case RULE_MISS_RANK8:\n"
                "        return miss_rank8;\n"
            )
            extra_args.append("int64_t miss_rank8")
            call_args.append("miss_rank8")
            feature_rows.append(
                "        double rank_second = -INFINITY;\n"
                "        for (int c = 0; c < NUM_CLASSES; ++c) {\n"
                "            if (c != arg && row[c] > rank_second) { rank_second = row[c]; }\n"
                "        }\n"
                "        int64_t miss_rank8 = clamp_i64(\n"
                "            (int64_t)(rank_second * 8.0 / one_minus), 0, 7);\n"
            )
        else:  # pragma: no cover - guarded by CHART_SETS
            raise Oc2Error(f"native port does not know chart {chart}")

    text = replace_first(
        text,
        "    RULE_PATCH192_ONLY            /* patch192_only, ddm_lb1 */\n};",
        "    RULE_PATCH192_ONLY,           /* patch192_only, ddm_lb1 */\n"
        + ",\n".join(enum_rows)
        + "\n};",
        label="native rule enum",
    )
    text = replace_once(
        text,
        "    RULE_PATCH192_ONLY            /* patch192_only, ddm_lb1 */\n};",
        "    RULE_PATCH192_ONLY,           /* patch192_only, ddm_lb1 */\n"
        + ",\n".join(rule_rows)
        + "\n};",
        label="native family rules",
    )
    text = replace_once(
        text,
        "    192                                                          /* ddm_lb1 */\n};",
        "    192,                                                         /* ddm_lb1 */\n"
        + ",\n".join(size_rows)
        + "\n};",
        label="native family sizes",
    )
    text = replace_first(
        text,
        "    /* ddm_gb1 + ddm_jt21 + ddm_lb1 */\n    0, 0, 0\n};",
        "    /* ddm_gb1 + ddm_jt21 + ddm_lb1 + ddm_oc2 */\n    0, 0, 0, "
        + ", ".join("0" for _ in charts)
        + "\n};",
        label="native count limits",
    )
    text = replace_once(
        text,
        "    /* ddm_gb1 + ddm_jt21 + ddm_lb1 */\n    0, 0, 0\n};",
        "    /* ddm_gb1 + ddm_jt21 + ddm_lb1 + ddm_oc2 */\n    0, 0, 0, "
        + ", ".join("0" for _ in charts)
        + "\n};",
        label="native initial weights",
    )
    old_signature = (
        "                                        int64_t spatial, int64_t spatial4, int64_t homog,\n"
        "                                        int64_t groupbin8, int64_t patch192)"
    )
    new_signature = (
        "                                        int64_t spatial, int64_t spatial4, int64_t homog,\n"
        "                                        int64_t groupbin8, int64_t patch192,\n"
        "                                        "
        + ", ".join(extra_args)
        + ")"
    )
    text = replace_once(text, old_signature, new_signature, label="native rule arguments")
    text = replace_once(
        text,
        "    case RULE_PATCH192_ONLY:\n        return patch192;\n    default:",
        "    case RULE_PATCH192_ONLY:\n        return patch192;\n"
        + "".join(cases)
        + "    default:",
        label="native rule cases",
    )
    text = replace_once(
        text,
        "        int64_t patch192 = (y / 32) * (WIDTH / 32) + (x / 32);\n"
        "        for (int pos = 0; pos < N_FAMILIES; ++pos) {",
        "        int64_t patch192 = (y / 32) * (WIDTH / 32) + (x / 32);\n"
        + "".join(feature_rows)
        + "        for (int pos = 0; pos < N_FAMILIES; ++pos) {",
        label="native chart features",
    )
    text = replace_once(
        text,
        "                                  run_f, boundary_f, spatial, spatial4, homog, groupbin8,\n"
        "                                  patch192);",
        "                                  run_f, boundary_f, spatial, spatial4, homog, groupbin8,\n"
        "                                  patch192, "
        + ", ".join(call_args)
        + ");",
        label="native rule call",
    )
    c_tmp = c_path.with_suffix(".c.partial")
    c_tmp.write_text(text)
    os.replace(c_tmp, c_path)

    gate_before = file_fact(gate_path)
    gate = replace_once(
        gate_path.read_text(),
        '        "patch192_only",\n',
        '        "patch192_only",\n'
        + "".join(f'        "{chart}",\n' for chart in charts),
        label="ordered native gate members",
    )
    gate_tmp = gate_path.with_suffix(".py.partial")
    gate_tmp.write_text(gate)
    os.replace(gate_tmp, gate_path)

    inflate = runtime / "inflate.sh"
    inflate_before = file_fact(inflate)
    inflate_text = inflate.read_text()
    if PYTHON_CORRECTOR_SELECTION in inflate_text:
        inflate_text = replace_once(
            inflate_text,
            PYTHON_CORRECTOR_SELECTION,
            NATIVE_CORRECTOR_BUILD,
            label="native inflate selection",
        )
        inflate_tmp = inflate.with_suffix(".sh.partial")
        inflate_tmp.write_text(inflate_text)
        inflate_tmp.chmod(inflate.stat().st_mode)
        os.replace(inflate_tmp, inflate)

    binary = MEASUREMENT / f"f26_corrector_native_oc2_{'_'.join(charts)}.dylib"
    compile_run = subprocess.run(
        [
            os.environ.get("CC", "cc"),
            "-O3",
            "-std=c11",
            "-shared",
            "-fPIC",
            "-ffp-contract=off",
            "-fno-fast-math",
            "-Wall",
            "-Wextra",
            str(c_path),
            "-lm",
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if compile_run.returncode != 0 or compile_run.stderr.strip():
        raise Oc2Error(
            f"native compile refused rc={compile_run.returncode}: {compile_run.stderr}"
        )
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0,{str(runtime)!r}); "
                "from runtime.native_free_corrector import assert_config_matches; "
                "assert_config_matches(); print('CONFIG_MATCH')"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "CONFIG_MATCH":
        raise Oc2Error(f"native config gate refused:\n{probe.stdout}\n{probe.stderr}")
    return {
        "source_generation": 22,
        "source_c": file_fact(source_c),
        "c_before": c_before,
        "c_after": file_fact(c_path),
        "gate_before": gate_before,
        "gate_after": file_fact(gate_path),
        "inflate_before": inflate_before,
        "inflate_after": file_fact(inflate),
        "compiled_library": file_fact(binary),
        "compile_argv": compile_run.args,
        "config_gate": "PASS",
    }


def stage_native(set_name: str) -> dict[str, Any]:
    if set_name not in CHART_SETS:
        raise Oc2Error(f"unknown chart set {set_name!r}")
    charts = CHART_SETS[set_name]
    if set_name == "rank":
        adjudication_path = MEASUREMENT / "RANK_ADJUDICATION.json"
        if not adjudication_path.is_file():
            raise Oc2Error("rank native port requires exact adjudication first")
        adjudication = json.loads(adjudication_path.read_text())
        if adjudication.get("disposition") != "ADMITTED__SEAL_FIRE_ORDER_OWED":
            raise Oc2Error("rank chart did not clear the 30 B native-port admission bar")
        candidate = RANK_AFTER_STORE / "retained" / "candidate_oc2_rank_after.zip"
        receipt_path = RANK_AFTER_STORE / "retained" / "S1_encode_oc2_rank_after.json"
        source_runtime = RANK_AFTER_STORE / "runtime_candidate"
    else:
        candidate = RETAINED / f"candidate_{tag_for_set(set_name)}.zip"
        receipt_path = RETAINED / f"S1_encode_{tag_for_set(set_name)}.json"
        source_runtime = runtime_for_set(set_name)
    if not candidate.is_file() or not receipt_path.is_file():
        raise Oc2Error(f"candidate/receipt missing for native port: {set_name}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("tokens_changed") != 0 or not receipt.get("delta_trustworthy"):
        raise Oc2Error(f"candidate is not an admitted lossless exact row: {set_name}")
    runtime = STORE / f"runtime_native_{set_name}"
    staged = stage_runtime(source_runtime, candidate, runtime)
    native = patch_native_runtime(runtime, charts)
    verdict = check_pin_consistency(runtime)
    if verdict.verdict != CONSISTENT:
        raise Oc2Error(f"native runtime pin refusal: {verdict.summary()}")
    payload = {
        "schema": "ddm_oc2_native_port.v1",
        "axis": "[macOS-CPU native corrector build / no score claim]",
        "score_claim": False,
        "promotable": False,
        "set": set_name,
        "charts": list(charts),
        "candidate": file_fact(candidate),
        "receipt": file_fact(receipt_path),
        "runtime": staged,
        "native": native,
        "pin_consistency": verdict.verdict,
    }
    atomic_json(MEASUREMENT / f"NATIVE_PORT_{set_name}.json", payload)
    return payload


def stage_identity(set_name: str) -> dict[str, Any]:
    runtime = STORE / f"runtime_native_{set_name}"
    candidate = runtime / "archive.zip"
    if not candidate.is_file():
        raise Oc2Error(f"native runtime absent for identity: {runtime}")
    root = STORE / f"identity_native_{set_name}"
    data = root / "data"
    out = root / "out"
    data.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(candidate, "r") as archive:
        if archive.namelist() != ["p"]:
            raise Oc2Error(f"candidate archive shape changed: {archive.namelist()}")
        member = archive.read("p")
    p_path = data / "p"
    partial = p_path.with_suffix(".partial")
    partial.write_bytes(member)
    os.replace(partial, p_path)
    command = [str(runtime / "inflate.sh"), str(data), str(out), "upstream/public_test_video_names.txt"]
    env = dict(os.environ)
    env["PATH"] = str(REPO / ".venv" / "bin") + os.pathsep + env.get("PATH", "")
    started = time.time()
    run = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, check=False)
    raw = out / "0.raw"
    token_checkpoint = out / ".f26_decode_checkpoints" / "tokens_cpu_stage_complete.u8"
    if run.returncode != 0 or not raw.is_file() or not token_checkpoint.is_file():
        raise Oc2Error(f"native identity failed rc={run.returncode}: {run.stderr[-4000:]}")
    raw_fact = file_fact(raw)
    token_fact = file_fact(token_checkpoint)
    expected_raw = json.loads(LB1_NATIVE_IDENTITY.read_text())["result"]["raw"]["sha256"]
    identical = raw_fact["sha256"] == expected_raw and token_fact["sha256"] == PINS["tokens"][2]
    payload = {
        "schema": "ddm_oc2_native_identity.v1",
        "axis": "[macOS-CPU full receiver identity / no score claim]",
        "score_claim": False,
        "promotable": False,
        "set": set_name,
        "candidate": file_fact(candidate),
        "command": command,
        "returncode": run.returncode,
        "elapsed_seconds": time.time() - started,
        "raw": raw_fact,
        "expected_lb1_raw_sha256": expected_raw,
        "decoded_tokens": token_fact,
        "byte_identical": identical,
        "stdout_tail": run.stdout[-4000:],
        "stderr_tail": run.stderr[-4000:],
        "payload_retained": True,
    }
    atomic_json(MEASUREMENT / f"NATIVE_IDENTITY_{set_name}.json", payload)
    if not identical:
        raise Oc2Error("native decode completed but did not reproduce lb1 raw/token identity")
    return payload


def stage_manifest() -> dict[str, Any]:
    entries = []
    manifest_path = MEASUREMENT / "MANIFEST.json"
    for path in sorted(STORE.rglob("*")):
        if (
            path.is_file()
            and path != manifest_path
            and not path.name.endswith(".partial")
        ):
            entries.append(
                {
                    "path": str(path.relative_to(STORE)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    payload = {
        "schema": "ddm_oc2_retained_manifest.v1",
        "root": str(STORE),
        "entries": entries,
        "total_bytes": sum(int(row["bytes"]) for row in entries),
    }
    atomic_json(manifest_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "preflight",
            "screen",
            "preregister-rank",
            "seed-rank-controls",
            "prepare-rank",
            "adjudicate-rank",
            "prepare-control",
            "prepare",
            "native",
            "identity",
            "manifest",
        ),
    )
    parser.add_argument("--set", choices=tuple(CHART_SETS))
    parser.add_argument("--base", choices=("solo", "after"))
    parser.add_argument("--seed", type=int, default=20260829)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "preflight":
        payload = stage_preflight()
    elif args.stage == "screen":
        payload = stage_screen(args.seed)
    elif args.stage == "preregister-rank":
        payload = stage_preregister_rank()
    elif args.stage == "seed-rank-controls":
        payload = stage_seed_rank_controls()
    elif args.stage == "prepare-rank":
        if args.base is None:
            raise Oc2Error("--base is required for --stage prepare-rank")
        payload = stage_prepare_rank(args.base)
    elif args.stage == "adjudicate-rank":
        payload = stage_adjudicate_rank()
    elif args.stage == "prepare-control":
        payload = stage_prepare_control()
    elif args.stage in {"prepare", "native", "identity"}:
        if args.set is None:
            raise Oc2Error(f"--set is required for --stage {args.stage}")
        payload = {
            "prepare": stage_prepare,
            "native": stage_native,
            "identity": stage_identity,
        }[args.stage](args.set)
    else:
        payload = stage_manifest()
    print(json.dumps(payload, indent=2, sort_keys=True)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
