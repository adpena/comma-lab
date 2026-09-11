"""ddm_ren1: can the shipped 12-coefficient carrier ABSORB a render change?

This is the adversarial half of the step-0 probe, and it exists to attack this
arm's own conclusion rather than to confirm it.

The refusal in `.omx/research/ddm_ren1_renderer_provenance_and_refit_in_place_20260911.md`
rests on `ddm_pr1`'s n600 post-re-solve coupling `k_post = 13.82`.  The honest
objection is that `k_post` belongs to a DIRECTION, not to the renderer: pr1
measured one candidate, and a refit steered away from the pose-expensive spectrum
might be cheaper.  The step-0 probe measures, at n600, that a half-LSB render
change costs far more Pose in the SMOOTH spectrum than in the NOISE spectrum.  So
the case most likely to overturn the refusal is the NOISE arm, and that is the
one this producer re-solves.

The question, stated so it can come out either way: **after the terminal carrier
re-solve, does a perturbed frame 1 get back to the shipped d_pose?**  If it does,
the pose wall is a property of the spectrum and a refit could be steered under it,
and the refusal must be withdrawn.  If it does not, the wall is measured here on
this object instead of transferred from pr1.

The solver is the canonical one, unforked: `up2.solve_pair_realized` — uncapped
greedy descent on the REALIZED objective, every candidate rendered through the
exact receiver path and scored by the frozen CPU PoseNet.  The only thing this
producer supplies is a frame-1 source: a shim that renders the treatment's frame 1
on demand in place of the shipped raw raster.

Sampling is a SEEDED RANDOM subsample through `up2.select_pairs`, never a prefix —
the pose axis is where prefix bias is anti-conservative (2.54-4.21x harder), so a
prefix here would be the false-negative shape.  A subsample is a BRACKET, not an
n600 verdict, and is labelled as one.

Axis: `[macOS-CPU advisory, up2 realized solver, DALI GT lineage]`.  No score
claim, no promotion, no archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "src")]

import ddm_jg1_seg_solve as jg1
import ddm_ren1_step0_probe as probe
import ddm_up2_shipping_pose_solve as up2
import numpy as np

AXIS = "[macOS-CPU advisory, up2 realized solver, DALI GT lineage]"
STORE = Path("/Volumes/VertigoDataTier/pact/ddm_ren1/resolve_absorption")
RESERVE_BYTES = 40 << 30
TREATMENTS = ("control", "noise_p05", "smooth_p05")


class Ren1ResolveError(RuntimeError):
    """Refusal raised by this producer.  Never downgraded to a warning."""


def retain(path: Path, payload: bytes) -> dict:
    if not path.resolve().is_relative_to(STORE.resolve()):
        raise Ren1ResolveError(f"write outside the ren1 resolve store: {path}")
    if shutil.disk_usage(STORE.parent).free < RESERVE_BYTES + len(payload):
        raise Ren1ResolveError("STORAGE_BLOCK: keep all payloads; free space below reserve")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return probe.fact(path)


class TreatmentFrames:
    """A ``raw``-shaped frame-1 source for ``up2.solve_pair_realized``.

    ``solve_pair_realized`` touches its ``raw`` argument exactly once, as
    ``raw[2 * index + 1]`` with ``index = np.array([pair])`` -- it regenerates
    frame 0 itself from the codes under test.  So the whole contract is: return
    the treatment's camera-resolution frame 1 for the requested odd indices.
    Rendering on demand (and caching one pair) keeps the 1.8 GB raster out of RAM
    and off the tier while staying bit-identical to the step-0 probe, which built
    its frames through the same two calls.
    """

    def __init__(self, semantic, tokens, field, treatment: str):
        self._semantic = semantic
        self._tokens = tokens
        self._field = field
        self._treatment = treatment
        self._cache: dict[int, np.ndarray] = {}

    def _frame_for_pair(self, pair: int) -> np.ndarray:
        cached = self._cache.get(pair)
        if cached is not None:
            return cached
        chunk = np.array([pair], dtype=np.int64)
        base = jg1.render_frame1(self._semantic, self._tokens[chunk], chunk)
        if self._treatment == "control":
            frames = base
        elif self._field is None:  # pragma: no cover - guarded by the caller
            raise Ren1ResolveError(f"treatment {self._treatment} needs a field")
        else:
            frames = probe.apply_perturbation(base, self._field)
        # One pair at a time: the solver walks pairs in order and never looks back,
        # so a single-entry cache is the whole working set.
        self._cache = {pair: frames}
        return frames

    def __getitem__(self, index):
        odd = np.asarray(index).reshape(-1)
        if odd.size != 1:
            raise Ren1ResolveError("frame-1 shim serves exactly one pair per call")
        value = int(odd[0])
        if value % 2 != 1:
            raise Ren1ResolveError(f"frame-1 shim was asked for even index {value}")
        return self._frame_for_pair((value - 1) // 2)


def run(args) -> int:
    import torch

    started = time.time()
    if args.treatment not in TREATMENTS:
        raise Ren1ResolveError(f"unknown treatment {args.treatment!r}")
    out = STORE / args.treatment
    out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)
    pointer = probe._bind_pointer()
    up2.enable_posenet_gradients()  # the solver's own module-level preparation
    semantic = jg1.load_semantic_renderer(
        archive_path=probe.POINTER_TREE / "archive.zip",
        runtime_dir=probe.POINTER_TREE / "runtime",
    )
    state = up2.load_carrier_state(probe.POINTER_TREE, verify_archive=False)
    posenet = up2.load_posenet("cpu")
    tokens = jg1.load_tokens(probe.TOKEN_FIELD)
    targets, lineage = up2.load_gt_poses(jg1.DEFAULT_GT_DALI)
    if lineage != up2.LINEAGE_DALI:
        raise Ren1ResolveError(f"pose GT lineage is {lineage}, not DALI")

    smooth_field, noise_field = probe.perturbation_fields(args.seed)
    field = {"control": None, "smooth_p05": smooth_field, "noise_p05": noise_field}[args.treatment]
    frames = TreatmentFrames(semantic, tokens, field, args.treatment)

    # SEEDED RANDOM, never a prefix: up2.select_pairs returns the full field at
    # >=600 and a seeded sample below it.
    pairs = up2.select_pairs(args.pairs, args.seed)
    binding = {
        "schema": "ddm_ren1_resolve_absorption.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "treatment": args.treatment,
        "pointer": pointer,
        "pointer_move": 48,
        "sample": {
            "pairs": len(pairs),
            "seed": args.seed,
            "shape": "seeded random via up2.select_pairs" if len(pairs) < jg1.N_PAIRS else "full n600",
            "is_verdict_grade": bool(len(pairs) >= jg1.N_PAIRS),
        },
        "producer": probe.fact(Path(__file__)),
        "probe_module": probe.fact(Path(probe.__file__)),
        "up2": probe.fact(Path(up2.__file__)),
        "jg1": probe.fact(Path(jg1.__file__)),
        "solver": "up2.solve_pair_realized, uncapped greedy realized descent, offsets (-2,-1,1,2)",
        "threads": args.threads,
    }
    retain(out / "INPUTS.json", (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode())

    rows_path = out / "rows.jsonl"
    done = {}
    if args.resume and rows_path.is_file():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[int(row["pair"])] = row
        print(json.dumps({"resumed_rows": len(done)}), flush=True)

    with rows_path.open("a", encoding="utf-8") as stream:
        for position, pair in enumerate(pairs):
            if int(pair) in done:
                continue
            row = up2.solve_pair_realized(
                posenet,
                state,
                frames,
                targets[int(pair)],
                int(pair),
                state.codes[int(pair)],
                max_passes=args.max_passes,
            )
            done[int(pair)] = row
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if position % 5 == 0 or position == len(pairs) - 1:
                finished = [done[int(p)] for p in pairs if int(p) in done]
                print(
                    json.dumps(
                        {
                            "treatment": args.treatment,
                            "done": len(finished),
                            "of": len(pairs),
                            "start_mean": float(np.mean([r["start_d_pose"] for r in finished])),
                            "final_mean": float(np.mean([r["final_d_pose"] for r in finished])),
                            "seconds": round(time.time() - started, 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    ordered = [done[int(p)] for p in pairs if int(p) in done]
    start_mean = float(np.mean([r["start_d_pose"] for r in ordered]))
    final_mean = float(np.mean([r["final_d_pose"] for r in ordered]))
    result = {
        **binding,
        "pairs_solved": len(ordered),
        "start_d_pose_mean": start_mean,
        "final_d_pose_mean": final_mean,
        "recovery_factor": start_mean / final_mean if final_mean > 0 else float("inf"),
        "pairs_improved": int(sum(1 for r in ordered if r["final_d_pose"] < r["start_d_pose"])),
        "all_converged": bool(all(r["converged"] for r in ordered)),
        "total_evaluations": int(sum(r["evaluations"] for r in ordered)),
        "total_changed_coordinates": int(sum(r["changed_coordinates"] for r in ordered)),
        "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        "elapsed_seconds": time.time() - started,
        "note": (
            "A subsample is a BRACKET, not an n600 verdict. Compare final_d_pose_mean "
            "against the SAME sample's control row, never against the n600 pointer value."
        ),
    }
    retain(out / "RESULT.json", (json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
    print(json.dumps(result, indent=1, sort_keys=True), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", required=True, choices=TREATMENTS)
    parser.add_argument("--pairs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-passes", type=int, default=0, help="0 = run to convergence")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume_from is not None and args.resume_from.resolve() != STORE.resolve():
        raise SystemExit(f"wrong resume root: {args.resume_from}")
    STORE.mkdir(parents=True, exist_ok=True)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
