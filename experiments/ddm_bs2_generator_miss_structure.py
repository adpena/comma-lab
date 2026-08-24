"""ddm_bs2 — structure of HG1's generator-only miss against the dx2 categorical field.

Scorer-free, $0, local. Measures WHAT the 359,280 B unique-home residual is actually
paying for, so the born-small (residual-dropped) container can be priced on the axis
that decides it: how much of the miss sits where the frozen SegNet head is fragile.

Both inputs are retained payload; neither is written. Results + sha256 are persisted
(ALWAYS KEEP THE PAYLOAD): this script writes a JSON receipt and the per-frame miss
counts, not just scalars.

axis: [macOS-CPU advisory, scorer-free exact field measurement]
score_claim: false   promotion_eligible: false   pointer_moved: false
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# comma10k canonical order (CLAUDE.md, MEASURED 2026-06-27 — never luma-sorted).
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
N_FRAMES, HEIGHT, WIDTH = 600, 384, 512
N_POS = N_FRAMES * HEIGHT * WIDTH  # 117,964,800

SRC_FIELD = Path(
    "/Volumes/APDataStore/pact/ddm_dx2/r7/decode_r1/inflated/"
    ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
GEN_FIELD = Path(
    "/Volumes/APDataStore/pact/ddm_hg1_heterogeneous_analytic_generator_gate/"
    "retained/generators/generated_tokens.u8"
)
# Pins from ddm_hg1_heterogeneous_analytic_generator_gate_20260823.md:65,:100.
SRC_SHA = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
HG1_MISS_COUNT = 1_334_939  # HG1 :103


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def load_field(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size != N_POS:
        raise SystemExit(f"REFUSE: {path} has {raw.size} bytes, expected {N_POS}")
    return raw.reshape(N_FRAMES, HEIGHT, WIDTH)


def boundary_mask(field: np.ndarray) -> np.ndarray:
    """True where a position has a 4-neighbour of a different class (in-frame only).

    This is the codim-1 locus the frozen rank-4 head decides on; interior positions
    sit far from every class-pair hyperplane and are where argmax is stable.
    """
    out = np.zeros(field.shape, dtype=bool)
    out[:, :-1, :] |= field[:, :-1, :] != field[:, 1:, :]
    out[:, 1:, :] |= field[:, 1:, :] != field[:, :-1, :]
    out[:, :, :-1] |= field[:, :, :-1] != field[:, :, 1:]
    out[:, :, 1:] |= field[:, :, 1:] != field[:, :, :-1]
    return out


def main() -> int:
    for path in (SRC_FIELD, GEN_FIELD):
        if not path.exists():
            raise SystemExit(f"REFUSE: missing retained payload {path}")

    src_sha = sha256_file(SRC_FIELD)
    if src_sha != SRC_SHA:
        raise SystemExit(f"REFUSE: source field sha {src_sha} != pinned {SRC_SHA}")
    gen_sha = sha256_file(GEN_FIELD)

    src = load_field(SRC_FIELD)
    gen = load_field(GEN_FIELD)

    miss = src != gen
    n_miss = int(miss.sum())

    # --- CONTROL (positive): the field must not be trivially identical or disjoint.
    if n_miss == 0 or n_miss == N_POS:
        raise SystemExit(f"REFUSE: degenerate diff, n_miss={n_miss}")

    # --- Per-class areas and the confusion of the miss.
    src_counts = np.bincount(src.reshape(-1), minlength=5).astype(np.int64)
    confusion = np.zeros((5, 5), dtype=np.int64)
    src_m = src[miss]
    gen_m = gen[miss]
    np.add.at(confusion, (src_m.astype(np.intp), gen_m.astype(np.intp)), 1)
    if int(confusion.sum()) != n_miss:
        raise SystemExit("REFUSE: confusion does not sum to the miss count")

    # --- Where does the miss sit relative to the class boundary of the TRUE field?
    bnd = boundary_mask(src)
    n_bnd = int(bnd.sum())
    miss_on_bnd = int((miss & bnd).sum())

    # --- Per-frame concentration (Gini over frames).
    per_frame = miss.reshape(N_FRAMES, -1).sum(axis=1).astype(np.int64)
    sf = np.sort(per_frame)
    idx = np.arange(1, N_FRAMES + 1)
    gini = float((2 * (idx * sf).sum()) / (N_FRAMES * sf.sum()) - (N_FRAMES + 1) / N_FRAMES)

    per_class_miss = confusion.sum(axis=1)
    report = {
        "arm": "ddm_bs2",
        "axis": "[macOS-CPU advisory, scorer-free exact field measurement]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "source_field": {"path": str(SRC_FIELD), "sha256": src_sha, "bytes": N_POS},
        "generator_field": {"path": str(GEN_FIELD), "sha256": gen_sha, "bytes": N_POS},
        "n_positions": N_POS,
        "n_miss": n_miss,
        "miss_fraction": n_miss / N_POS,
        "hg1_cited_miss": HG1_MISS_COUNT,
        "reproduces_hg1": n_miss == HG1_MISS_COUNT,
        "class_names": list(CLASS_NAMES),
        "source_class_counts": src_counts.tolist(),
        "source_class_area_frac": (src_counts / N_POS).tolist(),
        "miss_by_true_class": per_class_miss.tolist(),
        "miss_share_by_true_class": (per_class_miss / n_miss).tolist(),
        "over_representation_vs_area": [
            float((per_class_miss[c] / n_miss) / (src_counts[c] / N_POS))
            if src_counts[c] else None
            for c in range(5)
        ],
        "confusion_true_to_generated": confusion.tolist(),
        "boundary_positions": n_bnd,
        "boundary_fraction_of_field": n_bnd / N_POS,
        "miss_on_boundary": miss_on_bnd,
        "miss_on_boundary_share": miss_on_bnd / n_miss,
        "boundary_enrichment": float((miss_on_bnd / n_miss) / (n_bnd / N_POS)),
        "per_frame_miss_gini": gini,
        "per_frame_miss_max": int(per_frame.max()),
        "per_frame_miss_min": int(per_frame.min()),
    }

    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bs2_generator_miss_structure.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    per_frame.astype(np.int64).tofile(out_dir / "bs2_per_frame_miss_counts.i64")

    print(json.dumps(report, indent=2))
    print(f"\nwrote receipts to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
