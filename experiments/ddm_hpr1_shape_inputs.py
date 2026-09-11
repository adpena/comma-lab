"""ddm_hpr1 -- build the caller-pinned trainer inputs for the HPAC SHAPE rungs.

Two artifacts, both derived from the move-45 object and nothing else:

``cache.pt``  the CURRENT n600 token field as the trainer's ``seg`` tensor, with the
              ``spatial_token_sha256`` metadata the JF1/CL2 cache contract requires.
              The field is the shipped one (sha ``a92e7d90...``), the same bytes whose
              re-encode reproduces move 45's tail, so the prior is retrained against
              exactly the symbols it will have to code.

``init.pt``   the SHIPPED integer prior as a float ``state_dict``, read out of move 45's
              own ``hpac`` member through the shipping loader.  The shape CONTROL and
              every shape rung warm-start from this one file, so the only difference
              between their runs is the geometry -- which is what makes the comparison a
              shape measurement rather than a training-budget measurement.

Axis ``[macOS-CPU advisory; derived from the shipped object]``; ``score_claim=false``.

Usage::

  python experiments/ddm_hpr1_shape_inputs.py --out-dir <arm store>/train_inputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from experiments.ddm_hpr1_shape_price import FIELD, FIELD_SHA, POINTER45_SHA, PROMOTED45, fact

ARM_STORES = ("/Volumes/VertigoDataTier/pact/ddm_hpr1", "/Volumes/APDataStore/pact/ddm_hpr1")
RESERVE_BYTES = 40 << 30


class InputsError(RuntimeError):
    """A trainer input cannot be derived from the shipped object as claimed."""


def build(out_dir: Path) -> dict:
    import torch

    from experiments import ddm_rlc1_run as landed

    if shutil.disk_usage(out_dir.parent).free < RESERVE_BYTES:
        raise InputsError("STORAGE_BLOCK: keep all existing evidence")
    if fact(FIELD)["sha256"] != FIELD_SHA:
        raise InputsError("field sha mismatch")
    out_dir.mkdir(parents=True, exist_ok=True)

    field = np.fromfile(FIELD, dtype=np.uint8).reshape(600, 384, 512)
    if int(field.min()) < 0 or int(field.max()) > 4:
        raise InputsError("token field leaves the five-class alphabet")
    seg = torch.from_numpy(np.ascontiguousarray(field))
    digest = hashlib.sha256(seg.contiguous().numpy().tobytes(order="C")).hexdigest()
    if digest != FIELD_SHA:
        raise InputsError("cache token sha does not equal the shipped field sha")
    cache_path = out_dir / "cache.pt"
    torch.save({"seg": seg, "spatial_token_sha256": digest}, cache_path)

    rx, renderer, _ = landed.io.load_runtime(PROMOTED45)
    parts = rx.read_residual_archive(PROMOTED45 / "archive.zip")
    if fact(PROMOTED45 / "archive.zip")["sha256"] != POINTER45_SHA:
        raise InputsError("promoted tree is not the move-45 archive")
    body = rx.materialize_ihs1(parts.hpac_blob, renderer)
    model = renderer.load_hpac(body, torch.device("cpu"))
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    if not state:
        raise InputsError("shipped prior produced an empty state dict")
    init_path = out_dir / "init.pt"
    torch.save({"state_dict": state, "source": "move45 shipped hpac member"}, init_path)

    payload = {
        "cache": fact(cache_path),
        "init": fact(init_path),
        "expected_cache_content_sha256": digest,
        "expected_init_sha256": fact(init_path)["sha256"],
        "field": fact(FIELD),
        "base_archive": fact(PROMOTED45 / "archive.zip"),
        "ihs1_body_sha256": hashlib.sha256(body).hexdigest(),
        "state_dict_keys": sorted(state),
        "producer": fact(Path(__file__)),
        "axis": "[macOS-CPU advisory; derived from the shipped object]",
        "score_claim": False,
    }
    (out_dir / "TRAIN_INPUTS.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not any(str(args.out_dir).startswith(root) for root in ARM_STORES):
        raise InputsError("out-dir must be inside this arm's store")
    payload = build(args.out_dir)
    print(json.dumps({k: v for k, v in payload.items() if k != "state_dict_keys"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
