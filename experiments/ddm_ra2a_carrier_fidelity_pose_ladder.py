#!/usr/bin/env python3
"""RA2-a: measure d_pose as a function of CARRIER FIDELITY -- the one number the
whole carrier-rank ladder is affordability-blind without.

WHY THIS EXISTS (the level error the round-2 review exposed)
------------------------------------------------------------
ra1 ranks rank-r carriers by greedy energy; ra1b ranks them exhaustively; a
rotated whitened basis beats both. All three minimise **Euclidean field MSE**
against the shipped carrier. The SCORED quantity is a PoseNet readout:

    d_pose = mean_k<6 ( PoseNet(gen_pair)_k - PoseNet(orig_pair)_k )^2

Euclidean-optimal is not PoseNet-optimal. Every rung of that ladder is optimal
in a metric nobody scores. CLAUDE.md's standing S-geometry-pullback law (P0
triple, task #974) names exactly this: pull every Euclidean site back to the
scored geometry. This tool takes the first measurement in the RIGHT metric.

WHAT IT MEASURES
----------------
The carrier renders frame_0 ONLY (inflate.py:645-673: output[2i] = slave =
einsum(coeff, basis) -> bicubic; output[2i+1] = master = semantic(tokens)).
SegNet reads x[:, -1] = frame_1 = the master, so the carrier is SEG-INVISIBLE by
construction and is a PURE POSE ACTUATOR. Therefore a carrier perturbation moves
d_pose and NOTHING ELSE in the distortion terms -- the cleanest actuator on the
vehicle.

The probe scales / rank-reduces the decoded coefficients and re-renders through
the SHIPPED receiver, then scores through upstream/evaluate.py itself. No
surrogate receiver, no re-rolled GT decode (upstream owns both sides), no MLX
PoseNet (measured 0.55% rel drift -- pk4's banked law; CPU is the authority).

THE FIRST POINT IS alpha=0, NOT rank 1
--------------------------------------
The ra2 charter's ladder starts at rank 1. alpha=0 -- carrier coefficients set
to zero, i.e. the carrier DELETED -- is cheaper than every rung, saves the whole
22,161 B stream (rate credit 0.01475 S = 154% of the remaining gap), and bounds
the entire curve from the far end in ONE decode. Measuring the extreme first and
bracketing inward is strictly better than walking the ladder blind. That omission
was mine; this tool fixes it.

Note alpha=0 is a MEASUREMENT, not a shippable archive: the shipped receiver
hard-gates the carrier body at 22,183 B (AMENDMENT 2 C5), so any adopted rank
cut needs the container port. The port does not change d_pose -- it changes
which bytes carry the same coefficients -- so the affordability question is
answerable BEFORE the port is built. That is the point.

DECISION RULE (needs no gate literal)
-------------------------------------
pose term = sqrt(10 * d_pose), so a rate credit R is paid for iff

    d_pose(alpha) / d_pose(base)  <  ((POSE + R) / POSE)^2

with POSE = 0.0082945765 on the contest-CUDA frontier. This probe runs on the
ADVISORY axis, where the base d_pose is ~21.4x larger (CPU-vs-CUDA pose axis,
measured); so the RATIO is the transferable quantity and the absolute advisory
d_pose is NOT a score. Report the ratio; let a dual-axis T4 row decide adoption.

Axis: [macOS-CPU advisory]. score_claim=False. promotable=False.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
GEN = Path(
    "/Volumes/APDataStore/pact/ddm_mp2_mixed_precision_receiver_close_20260815"
    "/generations/hv1_base_control"
)
DEFAULT_OUTPUT = Path("/Volumes/APDataStore/pact/ddm_ra2a_carrier_pose_ladder_20260816")

# Frontier custody (AMENDMENT 2 / charter head). Re-verified, never assumed.
ARCHIVE_SHA = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
ARCHIVE_BYTES = 182_759
CARRIER_BR_BYTES = 22_161  # the REAL stream (AMENDMENT 2 C3), not the 22,278 rebaseline
POSE_CUDA = 0.0082945765
S_PER_BYTE = 25.0 / 37_545_489.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_receiver(rc64_library: Path):
    """Import the SHIPPED inflate.py as a module (same bytes the contest runs)."""
    os.environ["CPR1_RC64_LIBRARY"] = str(rc64_library)
    # inflate.py:16 imports `runtime.f26_inflate` relative to its own tree.
    tree = REPO / "src/tac/pr130_runtime/fx1_runtime_tree"
    for extra in (str(tree), str(GEN), str(GEN / "cpr1")):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    # The CPR1/semantic-pose receiver (split_payload / unpack_semantic_pose /
    # render_video) is the repo copy; GEN/inflate.py is the F26 outer wrapper.
    spec = importlib.util.spec_from_file_location(
        "hv1_inflate", REPO / "src/tac/pr130_runtime/fx1_runtime_tree/inflate.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["hv1_inflate"] = module
    spec.loader.exec_module(module)
    return module


def build_rc64(build_dir: Path) -> Path:
    """Compile the receiver's range-coder backend exactly as inflate.sh does."""
    out = build_dir / "rc64_backend.so"
    subprocess.run(
        [
            os.environ.get("CC", "cc"), "-O3", "-std=c11", "-shared", "-fPIC",
            str(GEN / "runtime/entropy/rc64_backend.c"), "-o", str(out),
        ],
        check=True,
    )
    return out


def transform_coeff(coeff: torch.Tensor, mode: str, param: float, gram: np.ndarray):
    """Return (modified_coeff, description). Pure function of the decoded coeff."""
    if mode == "alpha":
        return coeff * float(param), f"alpha={param}"
    if mode == "rank":
        # Exhaustive-optimal keep set in the EUCLIDEAN metric, least-squares refit
        # (ra1b). Deliberately the incumbent selection: this probe measures what
        # the CURRENT ladder costs in the SCORED metric, so the pose-metric
        # successor has an honest comparison point.
        import itertools

        r = int(param)
        c = coeff.double().numpy()
        best_keep, best_mse = None, None
        for subset in itertools.combinations(range(coeff.shape[1]), r):
            keep = np.asarray(subset)
            grr = gram[np.ix_(keep, keep)]
            grc = gram[np.ix_(keep, np.arange(coeff.shape[1]))]
            refit = np.linalg.lstsq(grr, grc @ c.T, rcond=None)[0].T
            approx = np.zeros_like(c)
            approx[:, keep] = refit
            delta = c - approx
            mse = float(np.einsum("bi,ij,bj->b", delta, gram, delta).mean())
            if best_mse is None or mse < best_mse:
                best_mse, best_keep, best_refit = mse, keep, refit
        approx = np.zeros_like(c)
        approx[:, best_keep] = best_refit
        return (
            torch.from_numpy(approx).to(coeff.dtype),
            f"rank={r} keep={best_keep.tolist()}",
        )
    raise SystemExit(f"unknown mode {mode!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("alpha", "rank"), default="alpha")
    parser.add_argument(
        "--param", type=float, default=0.0,
        help="alpha scale (mode=alpha) or rank r (mode=rank). Default 0.0 = carrier DELETED.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()
    started = time.time()

    label = f"{args.mode}{args.param:g}".replace(".", "p")
    out = args.output / label
    retained = out / "retained"
    retained.mkdir(parents=True, exist_ok=True)

    # --- custody: refuse on any drift, never assume ------------------------
    archive = GEN / "archive.zip"
    got_sha, got_bytes = sha256_file(archive), archive.stat().st_size
    if got_sha != ARCHIVE_SHA or got_bytes != ARCHIVE_BYTES:
        raise SystemExit(
            f"CUSTODY REFUSED: archive sha/bytes {got_sha[:16]}/{got_bytes} "
            f"!= pinned {ARCHIVE_SHA[:16]}/{ARCHIVE_BYTES}"
        )

    build_dir = Path(tempfile.mkdtemp(prefix="ra2a_build_"))
    try:
        receiver = load_receiver(build_rc64(build_dir))
        device = torch.device(args.device)

        # --- decode exactly as inflate.py main() does --------------------
        # The receiver reads member "p" from the EXTRACTED archive dir (inflate.py:717).
        import zipfile

        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            if names != ["p"]:
                raise SystemExit(f"unexpected archive members {names!r}; expected ['p']")
            payload = zf.read("p")
        parts = receiver.split_payload(payload)
        decoded = receiver.decode_models(parts.models, model_codec=parts.model_codec)
        raw = decoded.raw
        sem_len = int.from_bytes(raw[:4], "little")
        car_len = int.from_bytes(raw[4:8], "little")
        sp_bytes = 8 + sem_len + car_len
        semantic, basis, coeff = receiver.unpack_semantic_pose(raw[:sp_bytes])
        hpac = receiver.load_hpac(raw[sp_bytes:], device)
        tokens = receiver.decode_tokens(hpac, parts.tokens, device,
                                        token_codec=parts.token_codec)
        del hpac

        # Gram of the receiver's OWN normalized basis -- the exact geometry the
        # Euclidean ladder optimises, reproduced here so the comparison is fair.
        flat = receiver.normalized_basis(basis.double()).reshape(basis.shape[0], -1)
        gram = (flat @ flat.T).numpy() / flat.shape[1]

        coeff_mod, description = transform_coeff(coeff.cpu(), args.mode, args.param, gram)

        # --- render through the SHIPPED renderer -------------------------
        inflated = out / "inflated"
        inflated.mkdir(parents=True, exist_ok=True)
        destination = inflated / "0.raw"
        receiver.render_video(semantic, basis, coeff_mod.to(device), tokens,
                              destination, device)

        # --- score through UPSTREAM's own evaluate.py --------------------
        submission = out / "submission_dir"
        submission.mkdir(parents=True, exist_ok=True)
        if not (submission / "archive.zip").exists():
            shutil.copy2(archive, submission / "archive.zip")
        if (submission / "inflated").exists() or (submission / "inflated").is_symlink():
            (submission / "inflated").unlink()
        (submission / "inflated").symlink_to(inflated)

        proc = subprocess.run(
            [sys.executable, str(REPO / "upstream" / "evaluate.py"),
             "--submission-dir", str(submission),
             "--uncompressed-dir", str(REPO / "upstream" / "videos"),
             "--device", args.device],
            capture_output=True, text=True, cwd=REPO,
        )
        stdout = proc.stdout
        (retained / "evaluate_stdout.txt").write_text(stdout + "\n---STDERR---\n" + proc.stderr)
        d_pose = None
        for line in stdout.splitlines():
            if "Average PoseNet Distortion" in line:
                d_pose = float(line.split(":")[-1].strip())
        if proc.returncode != 0 or d_pose is None:
            raise SystemExit(
                f"EVALUATE FAILED rc={proc.returncode}; stdout tail:\n{stdout[-2000:]}"
            )

        # P0 ALWAYS KEEP THE PAYLOAD: the render is the payload. Persist its
        # sha + length; the bytes themselves are deterministically rebuildable
        # from (archive, mode, param) which is recorded here -- certify-or-block.
        raw_sha, raw_bytes = sha256_file(destination), destination.stat().st_size
        if not args.keep_raw:
            destination.unlink()

        receipt = {
            "schema": "ra2a_carrier_fidelity_pose_ladder.v1",
            "axis": "[macOS-CPU advisory]",
            "score_claim": False,
            "promotable": False,
            "mode": args.mode,
            "param": args.param,
            "transform": description,
            "d_pose_advisory": d_pose,
            "custody": {"archive_sha256": got_sha, "archive_bytes": got_bytes},
            "render_payload": {
                "path": str(destination), "sha256": raw_sha, "bytes": raw_bytes,
                "retained_on_disk": bool(args.keep_raw),
                "rebuild_command": (
                    f"{sys.executable} experiments/ddm_ra2a_carrier_fidelity_pose_ladder.py "
                    f"--mode {args.mode} --param {args.param} --keep-raw"
                ),
            },
            "affordance_reference": {
                "POSE_CUDA": POSE_CUDA,
                "carrier_stream_bytes": CARRIER_BR_BYTES,
                "note": (
                    "ratio d_pose(alpha)/d_pose(base) is the transferable quantity; "
                    "the advisory pose axis runs ~21.4x the CUDA one, so the "
                    "ABSOLUTE advisory d_pose is not a score and never transfers."
                ),
            },
            "elapsed_s": time.time() - started,
        }
        path = retained / "RA2A_POSE_LADDER_POINT.json"
        path.write_text(json.dumps(receipt, indent=2))
        print(f"\n{description}: d_pose_advisory = {d_pose:.10e}")
        print(f"receipt -> {path}")
        return 0
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
