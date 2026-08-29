#!/usr/bin/env python3
"""Prepare the lb1 jt21 + patch192 lossless joint re-encode.

This module does not estimate a byte delta.  It re-derives the two source
receipts, stages the exact jt21 archive behind a pin-consistent receiver, and
adds one receiver-derived ``patch192`` mixer family.  The physical byte verdict
is produced separately by ``experiments/ddm_jg2_tail_reencode.py`` over all 600
frames, with every stream, checkpoint, ledger, and archive retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tac.candidate_seal import CONSISTENT, check_pin_consistency

from experiments.ddm_fcd1_field_for_coder_diagonal import (
    NATIVE_CORRECTOR_BUILD,
    PYTHON_CORRECTOR_SELECTION,
    stage_runtime,
)


AP_ROOT = Path("/Volumes/APDataStore/pact")
VERTIGO_ROOT = Path("/Volumes/VertigoDataTier/pact")
STORE = AP_ROOT / "ddm_lb1_banked_lossless_joint_collect"
SOURCE_RUNTIME = AP_ROOT / "ddm_gb1_groupbin8_conditioning" / "runtime_joint21"
SOURCE_RETAINED = AP_ROOT / "ddm_gb1_groupbin8_conditioning" / "retained"
JT21_ARCHIVE = SOURCE_RETAINED / "candidate_gb1_joint21.zip"
GB1_ARCHIVE = SOURCE_RETAINED / "candidate_gb1_groupbin8_surprise.zip"
JT21_RECEIPT = SOURCE_RETAINED / "S1_encode_gb1_joint21.json"
TOKENS = (
    VERTIGO_ROOT
    / "ddm_to2_token_ordering_race"
    / "measurement_v1"
    / "retained"
    / "input"
    / "dx2_tokens_decoded.u8"
)
MI1_ROOT = AP_ROOT / "ddm_mi1_indicator_model_axis" / "measurement_v1"
MI1_LADDERS = tuple(
    MI1_ROOT / f"LADDER_seed{seed}.json" for seed in (20260824, 777, 31337)
)

BASE_RUNTIME = STORE / "runtime_base_jt21"
PATCHED_RUNTIME = STORE / "runtime_joint22_patch192"
MEASUREMENT = STORE / "measurement_v1"
NATIVE_RUNTIME = STORE / "runtime_candidate_native"
CANDIDATE_ARCHIVE = STORE / "retained" / "candidate_lb1_joint22_patch192.zip"
GB1_NATIVE_HOME = REPO / "runtime-rs" / "native" / "f26-corrector" / "gb1_20family"

PINS = {
    "jt21_archive": (
        180_192,
        "ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3",
    ),
    "gb1_archive": (
        180_215,
        "ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4",
    ),
    "tokens": (
        117_964_800,
        "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb",
    ),
    "jt21_receipt": (
        1_983,
        "a117480b2f79efff194ccf3c54520039579ea3f5efbc3f35cf881b9945d5ffe9",
    ),
}
MINIMUM_FREE_BYTES = 8 << 30
WIDTH = 512
HEIGHT = 384
PATCH = 32


class Lb1Error(RuntimeError):
    """A custody, source-receipt, or runtime-wiring refusal."""


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


def require_pin(path: Path, key: str) -> dict[str, Any]:
    expected_bytes, expected_sha = PINS[key]
    observed = file_fact(path)
    if observed["bytes"] != expected_bytes or observed["sha256"] != expected_sha:
        raise Lb1Error(
            f"{key} pin mismatch at {path}: "
            f"{observed['bytes']}/{observed['sha256']} != "
            f"{expected_bytes}/{expected_sha}"
        )
    return observed


def rederive_patch192() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in MI1_LADDERS:
        payload = json.loads(path.read_text())
        matches = [row for row in payload["rows"] if row.get("context") == "patch192"]
        if len(matches) != 1:
            raise Lb1Error(f"expected one patch192 row in {path}, found {len(matches)}")
        row = matches[0]
        if row.get("cells") != 192 or row.get("shipped_already") is not False:
            raise Lb1Error(f"patch192 source row changed shape in {path}: {row}")
        rows.append(
            {
                "receipt": file_fact(path),
                "seed": int(payload["seed"]),
                "heldout_gain_bytes": float(row["heldout_gain_bytes"]),
                "in_sample_gain_bytes": float(row["in_sample_gain_bytes"]),
                "max_abs_offset": float(row["max_abs_offset"]),
                "cells": int(row["cells"]),
            }
        )

    y, x = np.indices((HEIGHT, WIDTH), dtype=np.int64)
    patch192 = (y // PATCH) * (WIDTH // PATCH) + x // PATCH
    tile48 = (y // 64) * (WIDTH // 64) + x // 64
    subtile4 = ((y % 64) // PATCH) * 2 + (x % 64) // PATCH
    reconstructed = (
        ((tile48 // (WIDTH // 64)) * 2 + subtile4 // 2) * (WIDTH // PATCH)
        + (tile48 % (WIDTH // 64)) * 2
        + subtile4 % 2
    )
    if not np.array_equal(patch192, reconstructed):
        raise Lb1Error("patch192 is not reconstructible from (tile48, subtile4)")
    if int(patch192.min()) != 0 or int(patch192.max()) != 191:
        raise Lb1Error("patch192 does not span exactly 192 receiver-derived cells")

    return {
        "status": "SOURCE_REDERIVED",
        "source_axis": "[macOS-CPU advisory / held-out model ledger; not byte-closed]",
        "rows": rows,
        "selected_seed": 20260824,
        "selected_heldout_gain_bytes": rows[0]["heldout_gain_bytes"],
        "receiver_expression": "patch192 = (y // 32) * 16 + (x // 32)",
        "factorization_verified": (
            "patch row/col reconstruct exactly from tile48 row/col and subtile4 row/col"
        ),
        "receiver_derived": True,
        "stored_bytes": 0,
        "lineage_boundary": (
            "mi1 measured a held-out code-length gain on the dx2 field; only the "
            "full physical re-encode on the jt21/gb1 lineage can establish bytes"
        ),
    }


def rederive_jt21() -> dict[str, Any]:
    receipt_fact = require_pin(JT21_RECEIPT, "jt21_receipt")
    receipt = json.loads(JT21_RECEIPT.read_text())
    jt21 = require_pin(JT21_ARCHIVE, "jt21_archive")
    gb1 = require_pin(GB1_ARCHIVE, "gb1_archive")
    stream = receipt.get("stream", {})
    candidate = receipt.get("candidate_archive", {})
    checks = {
        "candidate_fact_matches_receipt": (
            candidate.get("bytes") == jt21["bytes"]
            and candidate.get("sha256") == jt21["sha256"]
        ),
        "tokens_unchanged": receipt.get("tokens_changed") == 0,
        "source_control_passed": receipt.get("control", {}).get("byte_identical") is True,
        "stream_delta_matches_archive_delta": (
            receipt.get("token_stream_delta_bytes") == receipt.get("archive_delta_bytes") == -176
        ),
        "retained_stream_present": (
            Path(str(stream.get("path", ""))).is_file()
            and Path(str(stream.get("path", ""))).stat().st_size == stream.get("bytes")
            and sha256_file(Path(str(stream.get("path", "")))) == stream.get("sha256")
        ),
    }
    if not all(checks.values()):
        raise Lb1Error(f"jt21 receipt re-derivation failed: {checks}")
    marginal = int(jt21["bytes"]) - int(gb1["bytes"])
    if marginal != -23:
        raise Lb1Error(f"jt21 marginal drifted: {marginal} B")
    return {
        "status": "SOURCE_REDERIVED",
        "source_axis": "[macOS-CPU advisory / scorer-free exact byte measurement]",
        "receipt": receipt_fact,
        "candidate_archive": jt21,
        "gb1_pointer_archive": gb1,
        "marginal_bytes_vs_gb1": marginal,
        "delta_S_rate_vs_gb1": marginal * 25.0 / 37_545_489.0,
        "tokens_changed": 0,
        "checks": checks,
    }


def stage_preflight() -> dict[str, Any]:
    STORE.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(STORE).free
    if free < MINIMUM_FREE_BYTES:
        raise Lb1Error(f"APDataStore free {free} B < {MINIMUM_FREE_BYTES} B")
    source_runtime_verdict = check_pin_consistency(SOURCE_RUNTIME)
    if source_runtime_verdict.verdict != CONSISTENT:
        raise Lb1Error(
            f"source runtime is pin-inconsistent: {source_runtime_verdict.summary()}"
        )
    receipt = {
        "schema": "ddm_lb1_preflight.v1",
        "axis": "[macOS-CPU advisory / scorer-free source re-derivation]",
        "score_claim": False,
        "storage": {
            "path": str(STORE),
            "free_bytes": free,
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
            "status": "PASS",
        },
        "inputs": {
            "tokens": require_pin(TOKENS, "tokens"),
            "source_runtime": str(SOURCE_RUNTIME),
            "source_runtime_pin_consistency": source_runtime_verdict.verdict,
        },
        "patch192": rederive_patch192(),
        "jt21": rederive_jt21(),
    }
    atomic_json(MEASUREMENT / "PREFLIGHT.json", receipt)
    return receipt


FEATURE_ANCHOR = '            "groupbin8": (\n'
FEATURE_PATCH = (
    '            "patch192": (\n'
    "                ((flat // WIDTH) // 32) * (WIDTH // 32)\n"
    "                + ((flat % WIDTH) // 32)\n"
    "            ),\n"
)
SPEC_ANCHOR = "    def groupbin8_only(f):\n"
SPEC_PATCH = (
    "    def patch192_only(f):\n"
    '        return f["patch192"]\n\n'
    "    specs.update({\"patch192_only\": (192, patch192_only)})\n\n"
)
MEMBER_ANCHOR = '        "cls_groupbin8",\n'


def patch_runtime(runtime: Path) -> dict[str, Any]:
    target = runtime / "runtime" / "fx2_model_axis_corrector.py"
    before = file_fact(target)
    text = target.read_text()
    if '"patch192_only"' not in text:
        for anchor, replacement in (
            (FEATURE_ANCHOR, FEATURE_PATCH + FEATURE_ANCHOR),
            (SPEC_ANCHOR, SPEC_PATCH + SPEC_ANCHOR),
            (MEMBER_ANCHOR, MEMBER_ANCHOR + '        "patch192_only",\n'),
        ):
            if text.count(anchor) != 1:
                raise Lb1Error(f"runtime patch anchor is not singular: {anchor!r}")
            text = text.replace(anchor, replacement, 1)
        temporary = target.with_suffix(".py.partial")
        temporary.write_text(text)
        os.replace(temporary, target)

    code = "\n".join(
        (
            "import json, sys",
            f"sys.path.insert(0, {str(runtime)!r})",
            "from runtime.fx2_model_axis_corrector import fx2_family_specs",
            "from runtime.free_corrector import SHIPPED_CONFIG",
            "specs = fx2_family_specs()",
            "print(json.dumps({",
            " 'member_in_pool': 'patch192_only' in specs,",
            " 'member_in_config': 'patch192_only' in SHIPPED_CONFIG['families'],",
            " 'member_cells': int(specs['patch192_only'][0]),",
            " 'family_count': len(SHIPPED_CONFIG['families']),",
            "}))",
        )
    )
    probe = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    if probe.returncode != 0:
        raise Lb1Error(f"patched runtime import failed:\n{probe.stderr}")
    wiring = json.loads(probe.stdout.strip().splitlines()[-1])
    expected = {
        "member_in_pool": True,
        "member_in_config": True,
        "member_cells": 192,
        "family_count": 22,
    }
    if wiring != expected:
        raise Lb1Error(f"patch192 wiring differs from expected: {wiring} != {expected}")
    return {"target_before": before, "target_after": file_fact(target), "wiring": wiring}


def replace_once(text: str, before: str, after: str, *, label: str) -> str:
    if text.count(before) != 1:
        raise Lb1Error(f"native port anchor {label!r} occurs {text.count(before)} times")
    return text.replace(before, after, 1)


def patch_native_runtime(runtime: Path) -> dict[str, Any]:
    """Port the git-custodied gb1 generation-20 C corrector to lb1 generation 22."""
    c_path = runtime / "runtime" / "f26_corrector_native.c"
    py_path = runtime / "runtime" / "native_free_corrector.py"
    inflate_path = runtime / "inflate.sh"
    shutil.copy2(GB1_NATIVE_HOME / c_path.name, c_path)
    shutil.copy2(GB1_NATIVE_HOME / py_path.name, py_path)

    c_before = file_fact(c_path)
    c_text = c_path.read_text()
    c_replacements = (
        (
            "#define N_FAMILIES 20 /* ddm_gb1: +groupbin8_surprise */",
            "#define N_FAMILIES 22 /* ddm_lb1: +cls_groupbin8 +patch192_only */",
            "family count",
        ),
        (
            "    /* ddm_gb1: decode-scan group conditioning. */\n"
            "    RULE_GROUPBIN8_SURPRISE\n};",
            "    /* ddm_gb1: decode-scan group conditioning. */\n"
            "    RULE_GROUPBIN8_SURPRISE,      /* groupbin8_surprise, ddm_gb1 */\n"
            "    RULE_CLS_GROUPBIN8,           /* cls_groupbin8, ddm_jt21 */\n"
            "    RULE_PATCH192_ONLY            /* patch192_only, ddm_lb1 */\n};",
            "rule enum",
        ),
        (
            "    RULE_GROUPBIN8_SURPRISE       /* groupbin8_surprise, ddm_gb1 */\n};",
            "    RULE_GROUPBIN8_SURPRISE,      /* groupbin8_surprise, ddm_gb1 */\n"
            "    RULE_CLS_GROUPBIN8,           /* cls_groupbin8, ddm_jt21 */\n"
            "    RULE_PATCH192_ONLY            /* patch192_only, ddm_lb1 */\n};",
            "family rules",
        ),
        (
            "    NUM_CLASSES * GROUP_BINS * U_BINS                            /* 2560, ddm_gb1 */\n};",
            "    NUM_CLASSES * GROUP_BINS * U_BINS,                           /* 2560, ddm_gb1 */\n"
            "    NUM_CLASSES * GROUP_BINS,                                    /* 40, ddm_jt21 */\n"
            "    192                                                          /* ddm_lb1 */\n};",
            "family sizes",
        ),
        (
            "static const int64_t FAMILY_COUNT_LIMIT[N_FAMILIES] = {\n"
            "    0, 0, 0, 0, 0, 0, 0, 0, 256, 4096, 256, 0, 0,\n"
            "    /* ddm_fx5: E1's six -- the two ``_fast256`` members carry the recency window. */\n"
            "    0, 0, 0, 0, 256, 256,\n"
            "    /* ddm_gb1 */\n    0\n};",
            "static const int64_t FAMILY_COUNT_LIMIT[N_FAMILIES] = {\n"
            "    0, 0, 0, 0, 0, 0, 0, 0, 256, 4096, 256, 0, 0,\n"
            "    /* ddm_fx5: E1's six -- the two ``_fast256`` members carry the recency window. */\n"
            "    0, 0, 0, 0, 256, 256,\n"
            "    /* ddm_gb1 + ddm_jt21 + ddm_lb1 */\n    0, 0, 0\n};",
            "count limits",
        ),
        (
            "static const int FAMILY_IS_SHIPPED_JOINT[N_FAMILIES] = {\n"
            "    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n"
            "    /* ddm_fx5: E1's six all start at weight 0, so the mixture still BEGINS at the\n"
            "     * incumbent law and the learner must earn every byte away from it. */\n"
            "    0, 0, 0, 0, 0, 0,\n"
            "    /* ddm_gb1 */\n    0\n};",
            "static const int FAMILY_IS_SHIPPED_JOINT[N_FAMILIES] = {\n"
            "    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n"
            "    /* ddm_fx5: E1's six all start at weight 0, so the mixture still BEGINS at the\n"
            "     * incumbent law and the learner must earn every byte away from it. */\n"
            "    0, 0, 0, 0, 0, 0,\n"
            "    /* ddm_gb1 + ddm_jt21 + ddm_lb1 */\n    0, 0, 0\n};",
            "initial weights",
        ),
        (
            "                                        int64_t spatial, int64_t spatial4, int64_t homog,\n"
            "                                        int64_t groupbin8)",
            "                                        int64_t spatial, int64_t spatial4, int64_t homog,\n"
            "                                        int64_t groupbin8, int64_t patch192)",
            "rule arguments",
        ),
        (
            "    case RULE_GROUPBIN8_SURPRISE:\n"
            "        return (cls * GROUP_BINS + groupbin8) * U_BINS + ubin;\n"
            "    default:",
            "    case RULE_GROUPBIN8_SURPRISE:\n"
            "        return (cls * GROUP_BINS + groupbin8) * U_BINS + ubin;\n"
            "    case RULE_CLS_GROUPBIN8:\n"
            "        return cls * GROUP_BINS + groupbin8;\n"
            "    case RULE_PATCH192_ONLY:\n"
            "        return patch192;\n"
            "    default:",
            "rule bodies",
        ),
        (
            "        int64_t groupbin8 = (((x % 64) + 2 * (y % 64)) * 8) / 190;\n"
            "        for (int pos = 0; pos < N_FAMILIES; ++pos) {",
            "        int64_t groupbin8 = (((x % 64) + 2 * (y % 64)) * 8) / 190;\n"
            "        int64_t patch192 = (y / 32) * (WIDTH / 32) + (x / 32);\n"
            "        for (int pos = 0; pos < N_FAMILIES; ++pos) {",
            "patch coordinate",
        ),
        (
            "                                  run_f, boundary_f, spatial, spatial4, homog, groupbin8);",
            "                                  run_f, boundary_f, spatial, spatial4, homog, groupbin8,\n"
            "                                  patch192);",
            "rule call",
        ),
    )
    for before, after, label in c_replacements:
        c_text = replace_once(c_text, before, after, label=label)
    c_tmp = c_path.with_suffix(".c.partial")
    c_tmp.write_text(c_text)
    os.replace(c_tmp, c_path)

    py_before = file_fact(py_path)
    py_text = replace_once(
        py_path.read_text(),
        '        "groupbin8_surprise",\n    ),',
        '        "groupbin8_surprise",\n'
        '        "cls_groupbin8",\n'
        '        "patch192_only",\n'
        "    ),",
        label="native expected families",
    )
    py_tmp = py_path.with_suffix(".py.partial")
    py_tmp.write_text(py_text)
    os.replace(py_tmp, py_path)

    inflate_before = file_fact(inflate_path)
    inflate_text = replace_once(
        inflate_path.read_text(),
        PYTHON_CORRECTOR_SELECTION,
        NATIVE_CORRECTOR_BUILD,
        label="native inflate selection",
    )
    inflate_tmp = inflate_path.with_suffix(".sh.partial")
    inflate_tmp.write_text(inflate_text)
    inflate_tmp.chmod(inflate_path.stat().st_mode)
    os.replace(inflate_tmp, inflate_path)

    binary = MEASUREMENT / "f26_corrector_native_lb1.dylib"
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
        raise Lb1Error(
            f"native compile refused rc={compile_run.returncode}: {compile_run.stderr}"
        )
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(runtime)!r}); "
                "from runtime.native_free_corrector import assert_config_matches; "
                "assert_config_matches(); print('CONFIG_MATCH')"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "CONFIG_MATCH":
        raise Lb1Error(f"native config gate refused:\n{probe.stdout}\n{probe.stderr}")
    return {
        "c_source_before": c_before,
        "c_source_after": file_fact(c_path),
        "python_gate_before": py_before,
        "python_gate_after": file_fact(py_path),
        "inflate_before": inflate_before,
        "inflate_after": file_fact(inflate_path),
        "compiled_library": file_fact(binary),
        "compile_argv": compile_run.args,
        "compile_stderr": compile_run.stderr,
        "config_gate": "PASS",
    }


def stage_native() -> dict[str, Any]:
    if not CANDIDATE_ARCHIVE.is_file():
        raise Lb1Error(f"candidate archive missing: {CANDIDATE_ARCHIVE}")
    candidate = file_fact(CANDIDATE_ARCHIVE)
    if candidate["bytes"] != 180_083 or candidate["sha256"] != (
        "5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9"
    ):
        raise Lb1Error(f"candidate custody pin changed: {candidate}")
    staged = stage_runtime(PATCHED_RUNTIME, CANDIDATE_ARCHIVE, NATIVE_RUNTIME)
    native = patch_native_runtime(NATIVE_RUNTIME)
    verdict = check_pin_consistency(NATIVE_RUNTIME)
    if verdict.verdict != CONSISTENT:
        raise Lb1Error(f"native candidate runtime pin refusal: {verdict.summary()}")
    receipt = {
        "schema": "ddm_lb1_native_port.v1",
        "axis": "[macOS-CPU native corrector build / no score claim]",
        "score_claim": False,
        "candidate_archive": candidate,
        "runtime": staged,
        "native_port": native,
        "pin_consistency": verdict.verdict,
        "derivation": (
            "git-custodied gb1 generation-20 native source plus exact jt21 cls_groupbin8 "
            "and lb1 patch192 rules"
        ),
    }
    atomic_json(MEASUREMENT / "NATIVE_PORT.json", receipt)
    return receipt


def stage_prepare() -> dict[str, Any]:
    preflight = stage_preflight()
    base = stage_runtime(SOURCE_RUNTIME, JT21_ARCHIVE, BASE_RUNTIME)
    if PATCHED_RUNTIME.exists():
        verdict = check_pin_consistency(PATCHED_RUNTIME)
        if verdict.verdict != CONSISTENT:
            raise Lb1Error(f"existing patched runtime pin refusal: {verdict.summary()}")
        resumed = True
    else:
        temporary = PATCHED_RUNTIME.with_name(PATCHED_RUNTIME.name + ".partial")
        if temporary.exists():
            raise Lb1Error(f"partial runtime exists; inspect before retry: {temporary}")
        shutil.copytree(BASE_RUNTIME, temporary, copy_function=shutil.copy2)
        os.replace(temporary, PATCHED_RUNTIME)
        resumed = False
    patch = patch_runtime(PATCHED_RUNTIME)
    patched_verdict = check_pin_consistency(PATCHED_RUNTIME)
    if patched_verdict.verdict != CONSISTENT:
        raise Lb1Error(f"patched runtime pin refusal: {patched_verdict.summary()}")
    archive = require_pin(PATCHED_RUNTIME / "archive.zip", "jt21_archive")
    receipt = {
        "schema": "ddm_lb1_runtime_prepare.v1",
        "axis": "[build artifact / no score claim]",
        "score_claim": False,
        "preflight_path": str(MEASUREMENT / "PREFLIGHT.json"),
        "preflight_sha256": sha256_file(MEASUREMENT / "PREFLIGHT.json"),
        "base_runtime": base,
        "patched_runtime": {
            "path": str(PATCHED_RUNTIME),
            "archive": archive,
            "pin_consistency": patched_verdict.verdict,
            "resumed": resumed,
            "patch": patch,
        },
        "joint_encode_command": [
            ".venv/bin/python",
            "experiments/ddm_jg2_tail_reencode.py",
            "--stage",
            "encode",
            "--store",
            str(STORE),
            "--runtime-root",
            str(PATCHED_RUNTIME),
            "--pointer-archive",
            str(JT21_ARCHIVE),
            "--expect-pointer-sha256",
            PINS["jt21_archive"][1],
            "--tokens",
            str(TOKENS),
            "--tag",
            "lb1_joint22_patch192",
            "--frames",
            "600",
            "--checkpoint-every",
            "25",
        ],
        "payload_retention": (
            "ddm_jg2 retains the stream, per-frame bit ledger, checkpoint, RC64 "
            "build, receipt, and candidate archive under this store"
        ),
    }
    atomic_json(MEASUREMENT / "RUNTIME_PREPARE.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=("preflight", "prepare", "native"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "preflight":
        receipt = stage_preflight()
    elif args.stage == "prepare":
        receipt = stage_prepare()
    else:
        receipt = stage_native()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
