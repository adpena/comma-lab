"""ddm_dpi1 -- rebuild the HPAC warm-start initializer WITH its bit-depth state.

THE DEFECT.  ``tools/train_ddm_cl1_hpac_capacity.py`` warm-starts every reference refit
from ``ddm_hpr1/train_inputs/init.pt``.  That file's ``state_dict`` carries ZERO
``*.bit_depth`` entries, and the trainer explicitly tolerates the omission
(``allowed_missing``), so ``enable_self_compression(model, init_bits=8.0)`` stands and
EVERY refit under the law starts its 517 row depths at 8 bits.  hpr1 measured the
consequence without naming the cause: retrained priors sit at mean row depth 5.888-5.890
against the shipped prior's 4.116.  This producer returns the lost state.

THE CHARTER'S SOURCE IS FALSIFIED, AND A BETTER ONE IS PROVEN HERE.  The charter names
``ddm_hv1_harvest_compose/retained/epoch_0634.pt`` as the source.  MEASURED: init.pt's own
``source`` field says ``move45 shipped hpac member`` -- it was never cut from epoch_0634 --
and epoch_0634's deployed depths average 4.2186 bits, not the shipped 4.1161.  The exact
ancestor is cl2's lambda=1.0 terminal EMA state: this producer PROVES, module by module,
that deploying that checkpoint's weights under its OWN bit depths reproduces init.pt's
weights, biases and exponents EXACTLY, and that the deployed depths reproduce the shipped
body's 517 row depths as an exact multiset.  Refuses if either proof fails.

Axis ``[macOS-CPU advisory; state restoration, scorer-free]``; ``score_claim=false``.

Usage::

  python experiments/ddm_dpi1_build_init_depths.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import torch
import torch.nn.functional as F

#: This arm's store.  Local disk is not a storage tier.
STORE = Path("/Volumes/VertigoDataTier/pact/ddm_dpi1")
#: The fail-closed free-space floor the sister HPAC producers hold.  Never lowered.
RESERVE_BYTES = 40 << 30

#: The law's warm start, weights only.
INIT = Path("/Volumes/VertigoDataTier/pact/ddm_hpr1/train_inputs/init.pt")
INIT_SHA = "cf41112788757f877d8d729d4bd7900772817cda8e06f6fd7fcb0129dc1f8383"
#: The charter's NAMED source, kept here as a measured comparison only.
E634 = Path("/Volumes/VertigoDataTier/pact/ddm_hv1_harvest_compose/retained/epoch_0634.pt")
E634_SHA = "5007beae7af7789758092f12f49096e13692e2e59850c85eb4642cd6fad147ec"
#: The MEASURED exact ancestor: cl2's lambda=1.0 terminal EMA state.
CL2 = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder"
    "/rungs/lambda_1p0/retained/terminal_checkpoint.pt"
)
#: The move-45 promoted tree, whose packed body carries the shipped row depths.
PROMOTED45 = Path("/Volumes/VertigoDataTier/pact/ddm_pc3_pose_carrier_curve/candidate/candidate_runtime")
POINTER45_SHA = "145e02e21f9a1cbc8276d1ecc34f0b9ae4762afa3fea7811e5836fee770ae60a"

WEIGHT_BOUND = 127.0
EXPONENT_MIN = -6
MAX_BITS = float(math.ceil(math.log2(2 * WEIGHT_BOUND + 1)))


class BuildError(RuntimeError):
    """A restoration input or invariant is not what the shipped object says it is."""


def fact(path: Path) -> dict:
    payload = Path(path).read_bytes()
    return {
        "path": str(Path(path)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def deployed_bits(bit_depth: torch.Tensor) -> torch.Tensor:
    """The depth the serializer actually writes for each row."""
    return F.relu(bit_depth.detach().float()).clamp(max=MAX_BITS).round()


def deployed_weight(weight: torch.Tensor, bit_depth: torch.Tensor) -> torch.Tensor:
    """``hpac_self_compress._quantized_weight`` with ``self_compress_deployed`` set.

    The mask multiply is omitted deliberately: it can only send values to zero, so an
    EXACT match without it is a strictly stronger statement than one with it.
    """
    bits = deployed_bits(bit_depth)
    radius = torch.pow(torch.tensor(2.0), bits - 1.0)
    view = (weight.shape[0],) + (1,) * (weight.ndim - 1)
    low = torch.maximum(-radius, torch.tensor(-WEIGHT_BOUND)).view(view)
    high = torch.minimum(radius - 1.0, torch.tensor(WEIGHT_BOUND)).view(view)
    return torch.round(torch.maximum(torch.minimum(weight.float(), high), low))


def histogram(depths) -> dict[str, int]:
    counts = Counter(int(value) for value in depths)
    return {str(bits): counts[bits] for bits in sorted(counts)}


def shipped_row_depths() -> tuple[list[int], dict]:
    """The 517 row depths the move-45 archive actually ships, read with the receiver."""
    from experiments import ddm_rlc1_run as landed

    archive = PROMOTED45 / "archive.zip"
    receipt = fact(archive)
    if receipt["sha256"] != POINTER45_SHA:
        raise BuildError(f"move-45 tree is {receipt['sha256']}, not the pointer this depth state belongs to")
    rx, renderer, _ = landed.io.load_runtime(PROMOTED45)
    from runtime import ihs2
    from runtime import rc2_hpac_semistatic_mixing as rc2

    parts = rx.read_residual_archive(archive)
    counts = list(ihs2.layout_from_runtime(renderer).row_counts)
    body = rx.materialize_ihs1(parts.hpac_blob, renderer)
    rows, depths = rc2.unpack_rows(body, counts)
    return [int(value) for value in depths], {
        "archive": receipt,
        "rows": len(rows),
        "values": int(sum(row.size for row in rows)),
        "body_bytes": len(body),
        "mean_row_depth_bits": float(sum(int(v) for v in depths) / len(depths)),
        "histogram": histogram(depths),
    }


def checkpoint_depth_view(path: Path, field: str) -> dict:
    """The deployed depth statistics a candidate source would restore."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload[field]
    names = sorted(name for name in state if name.endswith(".bit_depth"))
    values: list[int] = []
    for name in names:
        values.extend(int(v) for v in deployed_bits(state[name]).to(torch.int64).tolist())
    return {
        "file": fact(path),
        "field": field,
        "epoch": payload.get("epoch"),
        "bit_depth_tensors": len(names),
        "bit_depth_names": names,
        "rows": len(values),
        "mean_row_depth_bits": (sum(values) / len(values)) if values else None,
        "histogram": histogram(values),
    }


def prove_exact_ancestor(init_state: dict, source_state: dict) -> dict:
    """Refuse unless deploying the source under its own depths IS init.pt."""
    modules = sorted({name.rsplit(".", 1)[0] for name in source_state if name.endswith(".bit_depth")})
    checks = []
    for module in modules:
        weight = source_state[f"{module}.weight"]
        depth = source_state[f"{module}.bit_depth"]
        produced = deployed_weight(weight, depth)
        target = init_state[f"{module}.weight"].float()
        bias_ok = torch.equal(
            torch.round(source_state[f"{module}.bias"].float().clamp(-32768, 32767)),
            init_state[f"{module}.bias"].float(),
        )
        exponent_ok = torch.equal(
            torch.round(source_state[f"{module}.exponent"].float().clamp(EXPONENT_MIN, 0)),
            init_state[f"{module}.exponent"].float(),
        )
        weight_ok = torch.equal(produced, target)
        checks.append(
            {
                "module": module,
                "weight_exact": weight_ok,
                "bias_exact": bias_ok,
                "exponent_exact": exponent_ok,
                "rows": int(depth.numel()),
            }
        )
        if not (weight_ok and bias_ok and exponent_ok):
            raise BuildError(
                f"{module}: deploying the candidate source under its own depths does not reproduce init.pt"
            )
    frame_ok = torch.equal(
        torch.round(source_state["frame_embed.weight"].float()),
        init_state["frame_embed.weight"].float(),
    )
    if not frame_ok:
        raise BuildError("frame_embed.weight does not reproduce init.pt")
    return {"modules": checks, "frame_embed_exact": frame_ok, "all_exact": True}


def trainer_epoch_zero_histogram(init_depths: Path, intake_code: Path) -> dict:
    """The trainer's OWN ``bit_depth_histogram``, on a model built the trainer's way."""
    code = str(intake_code)
    if code not in sys.path:
        sys.path.insert(0, code)
    from hpac_integer import IntegerHPAC
    from hpac_self_compress import bit_depth_histogram, enable_self_compression, set_deployed_bit_depths

    model = IntegerHPAC(
        channels=64,
        patch=64,
        delta=2,
        frame_dim=8,
        norm_mode="none",
        activation="relu",
        use_frame_scale=True,
        weight_bound=127,
        activation_bound=127,
        use_weight_scales=True,
        weight_exponent_min=-6,
        use_spm=True,
        use_norm_gates=False,
    )
    enable_self_compression(model, 8.0)
    payload = torch.load(init_depths, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    missing = [name for name in incompatible.missing_keys]
    if incompatible.unexpected_keys or missing:
        raise BuildError(f"restored initializer is not a clean load: {incompatible}")
    set_deployed_bit_depths(model, True)
    counts = bit_depth_histogram(model)
    total = sum(counts.values())
    mean = sum(int(bits) * count for bits, count in counts.items()) / total
    return {
        "source": "tools/train_ddm_cl1_hpac_capacity.py's own bit_depth_histogram at epoch 0",
        "missing_keys": missing,
        "unexpected_keys": list(incompatible.unexpected_keys),
        "rows": total,
        "mean_row_depth_bits": mean,
        "histogram": counts,
    }


def main() -> int:
    if shutil.disk_usage(STORE.parent).free < RESERVE_BYTES + (1 << 20):
        raise BuildError("STORAGE_BLOCK: keep all existing evidence")
    init_receipt = fact(INIT)
    if init_receipt["sha256"] != INIT_SHA:
        raise BuildError("init.pt is not the pinned warm start")
    if fact(E634)["sha256"] != E634_SHA:
        raise BuildError("epoch_0634.pt is not the pinned source the charter names")

    init_payload = torch.load(INIT, map_location="cpu", weights_only=False)
    init_state = init_payload["state_dict"]
    present = sorted(name for name in init_state if name.endswith(".bit_depth"))
    if present:
        raise BuildError("init.pt already carries bit depths; the defect this cures is absent")

    cl2_payload = torch.load(CL2, map_location="cpu", weights_only=False)
    cl2_state = cl2_payload["state_dict"]
    ancestry = prove_exact_ancestor(init_state, cl2_state)

    shipped, shipped_receipt = shipped_row_depths()
    restored = {
        name: cl2_state[name].detach().clone().float()
        for name in sorted(cl2_state)
        if name.endswith(".bit_depth")
    }
    deployed = []
    for name in sorted(restored):
        deployed.extend(int(v) for v in deployed_bits(restored[name]).to(torch.int64).tolist())
    if Counter(deployed) != Counter(shipped):
        raise BuildError("restored depths do not reproduce the shipped row-depth multiset")
    restored_mean = sum(deployed) / len(deployed)
    if restored_mean != shipped_receipt["mean_row_depth_bits"]:
        raise BuildError("restored mean row depth differs from the shipped body's")

    out_dir = STORE / "train_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "init_depths.pt"
    if target.exists():
        raise BuildError(f"{target} already exists; immutable payloads are never overwritten")
    torch.save(
        {
            "source": (
                "init.pt (move45 shipped hpac member) + the 9 bit_depth tensors of cl2 "
                "lambda=1.0 terminal EMA state, PROVEN to be init.pt's exact ancestor"
            ),
            "state_dict": {**init_state, **restored},
        },
        target,
    )
    target.chmod(0o444)
    built = fact(target)

    epoch_zero = trainer_epoch_zero_histogram(
        target, Path("/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code")
    )
    if epoch_zero["histogram"] != shipped_receipt["histogram"]:
        raise BuildError("the trainer's own epoch-0 histogram is not the shipped histogram")

    receipt = {
        "schema": "ddm_dpi1_init_depths.v1",
        "axis": "[macOS-CPU advisory; state restoration, scorer-free]",
        "score_claim": False,
        "producer": fact(Path(__file__)),
        "defect": {
            "init_bit_depth_tensors": 0,
            "trainer_tolerates_missing": "allowed_missing in tools/train_ddm_cl1_hpac_capacity.py",
            "effective_epoch_0_depth_without_the_cure": 8.0,
        },
        "charter_source_correction": {
            "charter_claim": "init.pt is an EMA cut of epoch_0634.pt; take its 10 bit_depth tensors",
            "measured_init_source_field": init_payload.get("source"),
            "measured_e634_bit_depth_tensors": 9,
            "e634_state_dict": checkpoint_depth_view(E634, "state_dict"),
            "why_rejected": (
                "epoch_0634 is 60 epochs upstream of init.pt; its deployed depths do not "
                "reproduce the shipped 4.116 mean the charter's own acceptance test names"
            ),
        },
        "restored_from": checkpoint_depth_view(CL2, "state_dict"),
        "ancestry_proof": ancestry,
        "shipped_reference": shipped_receipt,
        "restored_depths": {
            "tensors": len(restored),
            "rows": len(deployed),
            "mean_row_depth_bits": restored_mean,
            "histogram": histogram(deployed),
            "multiset_equals_shipped": True,
        },
        "trainer_epoch_zero": epoch_zero,
        "init": init_receipt,
        "built": built,
        "cleanup": "KEEP; immutable 0444 payload on the SSD tier; 40 GiB reserve held",
    }
    receipt_path = out_dir / "INIT_DEPTHS.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
