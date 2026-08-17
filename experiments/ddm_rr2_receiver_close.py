"""ddm_rr2 - wire the free corrector into the receiver and prove the field.

``ddm_rr1`` §4 step 1 says: add the adaptive corrector to
``runtime/residual_archive.decode_production_tokens`` immediately after
``corrected = base_logits + parts.table.values[feature]``, apply the shift
before the probability table is handed to the coder, and update from the
decoded group.  Step 3 says: decode the rebuilt archive and assert the token
field reproduces sha ``9ba2e52b...`` bit-identically - that is the whole
distortion proof, because an identical field cannot move ``d_seg`` or
``d_pose``.

This script builds that candidate runtime tree and runs the proof.

THE ONE-FUNCTION-USED-TWICE CONTRACT.  ``runtime/free_corrector.py`` in the
candidate tree is a byte-identical copy of ``experiments/ddm_rr2_free_corrector.py``,
which is the module the encoder imports.  The receipt records both sha256s and
this script refuses to build if they differ, so the encoder and the receiver
cannot drift apart.

WHY THE RUNTIME MUST BE REGENERATED.  ``inflate.py`` pins the archive's sha256
and byte count, so a new archive necessarily needs its own runtime generation -
"same runtime tree, new archive" is structurally impossible on this family.
The pins are rewritten here by exact-string replacement with a fail-closed
occurrence check, never by a regex guess.

RATE NOTE.  ``inflate.py`` and everything under ``runtime/`` are unsized: the
score charges ``archive.zip`` bytes only.  The corrector is a generic algorithm
driven by already-decoded symbols, so adding it to the runtime costs zero
archive bytes and stores no video-derived content.

Axis: ``[macOS-CPU advisory]``.  No score claim; this proves byte identity of
the decoded field and closes the receiver, it is not an exact eval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE_TREE = Path(
    "/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/prepared/hv1_base_control"
)
# ddm_rr4: selectable corrector, default unchanged.  The receiver MUST ship the
# same module the encoder used; cross-check corrector_sha256 in the two receipts.
CORRECTOR_MODULE = os.environ.get("TAC_RR2_CORRECTOR_MODULE", "ddm_rr2_free_corrector")
CORRECTOR = REPO / "experiments" / f"{CORRECTOR_MODULE}.py"
STORE = Path("/Volumes/APDataStore/pact/ddm_rr2_encoder_build")
CANDIDATE_ARCHIVE = STORE / "retained" / "archive.zip"
TOKEN_FIELD_SHA = "9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52"
FRONTIER_ARCHIVE_SHA = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
FRONTIER_ARCHIVE_BYTES = 182_759
EXPECT_CORRECTED_SHA = "562ac652b372faa020d0fc5e2ed9b7b61625169e0f5c2041d4fe99196055b8c7"
EXPECT_CDF_SHA = "dd48843b021763e78524caf3dcd01e944045e7bd0ffd93b451dec83548f083b7"

IMPORT_ANCHOR = (
    "    from .hpac_inference import configure_cuda_reproducibility, optimize_sparse_evaluator\n"
)
SPARSE_ANCHOR = "    sparse = _sparse_class(code_dir)(model, runtime.EVAL_H, runtime.EVAL_W)\n"
BOUNDARY_ANCHOR = """                boundary = np.full(
                    runtime.EVAL_H * runtime.EVAL_W,
                    4,
                    dtype=np.uint8,
                )
"""
DECODE_ANCHOR = "                symbols = decoder.decode(probability).astype(np.int64)\n"
NATIVE_GUARD_ANCHOR = """    if token_decoder not in {"python", "native-hpac"}:
        raise InflationError("F26_TOKEN_DECODER must be 'python' or 'native-hpac'")
"""
NATIVE_GUARD = """    if token_decoder != "python":
        raise InflationError(
            "this generation wires the ddm_rr2 free probability corrector into the "
            "python token decoder only; the native-hpac path is unpatched and would "
            "decode a different field, so it is refused rather than trusted"
        )
"""
FRAME_END_ANCHOR = """            tokens[frame] = current[0].to(device="cpu", dtype=torch.uint8)
            previous = current
"""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_fact(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def progress(record: dict[str, object]) -> None:
    record = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record}
    with (STORE / "PROGRESS.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"patch anchor {label!r} appears {text.count(old)} times - refusing")
    return text.replace(old, new)


def patch_residual_archive(source: str) -> str:
    """Insert the corrector into the shipped decode loop, nothing else."""
    text = replace_once(
        source,
        IMPORT_ANCHOR,
        IMPORT_ANCHOR + "    from .free_corrector import FreeCorrector\n",
        "corrector import",
    )
    text = replace_once(
        text,
        SPARSE_ANCHOR,
        SPARSE_ANCHOR + "    corrector = FreeCorrector(runtime.EVAL_H * runtime.EVAL_W)\n",
        "corrector construction",
    )
    text = replace_once(
        text,
        BOUNDARY_ANCHOR,
        BOUNDARY_ANCHOR + "            corrector.begin_frame(boundary)\n",
        "frame boundary hand-off",
    )
    text = replace_once(
        text,
        DECODE_ANCHOR,
        "                state = corrector.group_state(\n"
        "                    probability,\n"
        "                    predicted,\n"
        "                    flat_positions,\n"
        "                )\n"
        "                symbols = decoder.decode(\n"
        "                    corrector.coding_row(state)\n"
        "                ).astype(np.int64)\n"
        "                corrector.observe(state, symbols)\n",
        "group decode",
    )
    return replace_once(
        text,
        FRAME_END_ANCHOR,
        '            tokens[frame] = current[0].to(device="cpu", dtype=torch.uint8)\n'
        "            corrector.end_frame(tokens[frame].numpy().reshape(-1))\n"
        "            previous = current\n",
        "frame end",
    )


def copy_tree(destination: Path) -> None:
    def ignore(directory: str, names: list[str]) -> set[str]:
        del directory
        return {name for name in names if name.startswith("._") or name == "__pycache__"}

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(SOURCE_TREE, destination, ignore=ignore)


def build_candidate(destination: Path) -> dict[str, object]:
    if sha256_file(SOURCE_TREE / "archive.zip") != FRONTIER_ARCHIVE_SHA:
        raise SystemExit("source runtime is not bound to the frontier archive - refusing")
    if not CANDIDATE_ARCHIVE.is_file():
        raise SystemExit("build the candidate archive first")
    candidate = file_fact(CANDIDATE_ARCHIVE)
    copy_tree(destination)

    shutil.copy2(CANDIDATE_ARCHIVE, destination / "archive.zip")
    shutil.copy2(CORRECTOR, destination / "runtime" / "free_corrector.py")
    if sha256_file(destination / "runtime" / "free_corrector.py") != sha256_file(CORRECTOR):
        raise SystemExit("corrector copy is not byte-identical - refusing")

    inflate = destination / "inflate.py"
    text = inflate.read_text()
    text = replace_once(
        text,
        f'ARCHIVE_SHA256 = "{FRONTIER_ARCHIVE_SHA}"\n',
        f'ARCHIVE_SHA256 = "{candidate["sha256"]}"\n',
        "archive sha pin",
    )
    text = replace_once(
        text,
        f"ARCHIVE_BYTES = {FRONTIER_ARCHIVE_BYTES:_}\n",
        f"ARCHIVE_BYTES = {int(candidate['bytes']):_}\n",
        "archive bytes pin",
    )
    inflate.write_text(text)

    residual = destination / "runtime" / "residual_archive.py"
    original = residual.read_text()
    residual.write_text(patch_residual_archive(original))

    # The shipping runtime exposes a second, native token decoder that this
    # arm did not patch.  Leaving it reachable would leave an unproven decode
    # path in a candidate, so it is refused rather than trusted.
    f26 = destination / "runtime" / "f26_inflate.py"
    f26_before = f26.read_text()
    f26.write_text(
        replace_once(f26_before, NATIVE_GUARD_ANCHOR, NATIVE_GUARD_ANCHOR + NATIVE_GUARD, "native decoder guard")
    )

    return {
        "candidate_runtime": str(destination),
        "archive": candidate,
        "corrector": {
            "repository_copy": file_fact(CORRECTOR),
            "runtime_copy": file_fact(destination / "runtime" / "free_corrector.py"),
            "one_function_used_twice": True,
        },
        "inflate": file_fact(inflate),
        "f26_inflate": {
            "native_hpac_path_refused": True,
            "after": file_fact(f26),
        },
        "residual_archive": {
            "before": {"bytes": len(original.encode()), "sha256": hashlib.sha256(original.encode()).hexdigest()},
            "after": file_fact(residual),
        },
    }


def run_parseback(candidate: Path, work: Path, expect_field_sha: str) -> dict[str, object]:
    data_dir = work / "extracted"
    output_dir = work / "inflated"
    for directory in (data_dir, output_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    with zipfile.ZipFile(candidate / "archive.zip") as archive:
        archive.extract("p", data_dir)
    file_list = work / "video_names.txt"
    file_list.write_text("0.mkv\n")

    environment = dict(os.environ)
    environment["PATH"] = f"{REPO / '.venv' / 'bin'}:{environment['PATH']}"
    environment["F26_TOKEN_DECODER"] = "python"
    log = work / "inflate_stdout.txt"
    started = time.perf_counter()
    with log.open("wb") as stream:
        completed = subprocess.run(
            ["bash", str(candidate / "inflate.sh"), str(data_dir), str(output_dir), str(file_list)],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started

    checkpoint = output_dir / ".f26_decode_checkpoints" / "tokens_cpu_stage_complete.json"
    receipt: dict[str, object] = {
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "log": file_fact(log),
        "checkpoint_present": checkpoint.is_file(),
    }
    # The whole inflated field, not only the token stage.  Two runs that agree
    # here have provably identical d_seg and d_pose, because the scorer sees
    # exactly these bytes.
    raw = output_dir / "0.raw"
    if raw.is_file():
        receipt["inflated_output"] = file_fact(raw)
    if checkpoint.is_file():
        payload = json.loads(checkpoint.read_text())
        decoder = payload.get("token_decoder", {})
        receipt["token_checkpoint"] = payload
        receipt["decoded_token_sha256"] = decoder.get("decoded_token_sha256")
        receipt["decoded_field_bit_identical"] = decoder.get("decoded_token_sha256") == expect_field_sha
        receipt["decode_runtime_seconds_token_stage"] = decoder.get("decode_runtime_seconds")
        receipt["base_probability_controls"] = {
            "corrected_quantized_logit_sha256_matches": decoder.get("corrected_quantized_logit_sha256")
            == EXPECT_CORRECTED_SHA,
            "corrected_cdf_input_sha256_matches": decoder.get("corrected_cdf_input_sha256") == EXPECT_CDF_SHA,
        }
    return receipt


def main() -> int:
    # ddm_rr4 fix: same SyntaxError as the encoder had - the declaration sat
    # after `default=STORE` had already read the name, so the committed
    # instrument did not compile.  Declaring it first is semantically identical.
    # ddm_rr4 fix: CANDIDATE_ARCHIVE was bound at import from the DEFAULT store,
    # so `--store` staged one store's runtime around another store's archive.
    # It is now rebound from the parsed argument.
    global STORE, CANDIDATE_ARCHIVE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("build", "parseback"))
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument(
        "--tree",
        type=Path,
        default=None,
        help="runtime tree to decode; defaults to the built candidate",
    )
    parser.add_argument(
        "--tag",
        default="candidate",
        help="names the parse-back work dir and receipt (use 'base' for the matched control)",
    )
    args = parser.parse_args()

    STORE = args.store
    CANDIDATE_ARCHIVE = args.store / "retained" / "archive.zip"
    candidate = args.tree if args.tree is not None else args.store / "candidate_runtime"
    work = args.store / f"parseback_{args.tag}" if args.tag != "candidate" else args.store / "parseback"
    work.mkdir(parents=True, exist_ok=True)

    result = (
        build_candidate(args.store / "candidate_runtime")
        if args.stage == "build"
        else run_parseback(candidate, work, TOKEN_FIELD_SHA)
    )
    result["runtime_tree"] = str(candidate)
    result["tag"] = args.tag
    result["stage"] = args.stage
    result["axis"] = "[macOS-CPU advisory]"
    result["score_claim"] = False
    result["promotable"] = False
    suffix = "" if args.tag == "candidate" else f"_{args.tag}"
    atomic_json(args.store / f"RESULT_receiver_{args.stage}{suffix}.json", result)
    progress({"stage": f"receiver-{args.stage}{suffix}", "event": "complete", "result": result})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
