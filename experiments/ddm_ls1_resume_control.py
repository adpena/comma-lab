#!/usr/bin/env python3
"""Actual-receiver restart control; prefix is an implementation test, not a verdict."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

import numpy as np
import torch

from experiments.ddm_ls1_shipped_surprise import ROOT, fact, record


def control(resume_from):
    """Restore frame25, re-decode to50, match every frequency and all restart state."""
    out = ROOT / "resume_control"
    if Path(resume_from).resolve() != (out / "checkpoints").resolve():
        raise ValueError("wrong isolated checkpoint root")
    out.mkdir(exist_ok=True)
    destination = Path(resume_from)
    destination.mkdir(exist_ok=True)
    source = ROOT / "receiver_checkpoints/stage_0025.npz"
    receipt = json.loads(source.with_suffix(".json").read_text())
    if fact(source)["sha256"] != receipt["sha256"]:
        raise ValueError("source checkpoint hash differs")
    copied = destination / source.name
    if not copied.exists():
        shutil.copyfile(source, copied)
    if fact(copied)["sha256"] != receipt["sha256"]:
        raise ValueError("copied checkpoint hash differs")
    receipt["path"] = str(copied)
    if not (destination / "LATEST.json").exists():
        record(destination / "LATEST.json", receipt)
    os.environ.update(
        CPR1_RC64_LIBRARY=str(ROOT / "native/librc64.so"),
        F26_CORRECTOR_NATIVE_LIBRARY=str(ROOT / "native/libcorrector.so"),
        RLC1_GEOMETRY_LIBRARY=str(ROOT / "native/geometry.dylib"),
        TC1_RECEIVER_CHECKPOINT_DIR=str(destination),
        TC1_RECEIVER_STOP_AFTER="50",
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    from experiments.ddm_jg2_tail_reencode import load_runtime

    residual, renderer, code_dir = load_runtime(ROOT / "runtime_copy")
    from runtime.entropy.rc64 import NativeDecoder
    from runtime.rlc1_mixer import LaneMixer
    from runtime.tc1_shared_mixer import frequencies

    state = {"frame": None, "positions": None, "checked": 0, "freq": None, "truth": None}
    old_coding, old_decode = LaneMixer.coding, NativeDecoder.decode

    def coding(self, rows, positions, plane, previous):
        result = old_coding(self, rows, positions, plane, previous)
        if state["frame"] != self.frame:
            path = ROOT / "rows" / f"frame_{self.frame:04d}.npz"
            if fact(path) != json.loads(path.with_suffix(".json").read_text()):
                raise ValueError("reference row changed")
            with np.load(path, allow_pickle=False) as data:
                state["freq"], state["truth"] = data["frequencies"], data["symbols"]
            state["frame"] = self.frame
        state["positions"] = positions
        return result

    def decode(self, probabilities):
        output = old_decode(self, probabilities)
        positions = state["positions"]
        np.testing.assert_array_equal(frequencies(probabilities), state["freq"][positions])
        np.testing.assert_array_equal(output, state["truth"][positions])
        state["checked"] += len(output)
        return output

    LaneMixer.coding, NativeDecoder.decode = coding, decode
    parts = residual.read_residual_archive(ROOT / "runtime_copy/archive.zip")
    try:
        residual.decode_production_tokens(parts, renderer, code_dir, torch.device("cpu"))
    except RuntimeError as error:
        if str(error) != "TC1_RECEIVER_STAGE_COMPLETE":
            raise
    else:
        raise ValueError("restart control failed to stop at frame50")
    if state["checked"] != 25 * 384 * 512:
        raise ValueError("restart comparison omitted symbols")
    resumed = destination / "stage_0050.npz"
    reference = ROOT / "receiver_checkpoints/stage_0050.npz"
    with np.load(resumed, allow_pickle=False) as a, np.load(reference, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError("restart state census mismatch")
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
        keys = a.files
    record(
        out / "RESULT.json",
        {
            "status": "PASS",
            "source": fact(Path(__file__)),
            "symbols_compared": state["checked"],
            "frequency_entries_compared": state["checked"] * 5,
            "state_arrays_compared": len(keys),
            "arrays": keys,
            "resumed": fact(resumed),
            "uninterrupted": fact(reference),
            "scope": "frames25..49 implementation restart control; atlas verdict remains full n600",
            "score_claim": False,
        },
    )
    print(json.dumps({"status": "PASS", "symbols_compared": state["checked"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", required=True)
    control(parser.parse_args().resume_from)
